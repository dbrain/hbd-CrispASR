#!/usr/bin/env python3
"""Re-pack a parakeet F16 GGUF to Q8_0.

Tensor strategy (matches what `quantize` ggml CLI tools normally pick):
  • 2D matmul weights (linear / embed / pointwise conv) → Q8_0 if last dim % 32 == 0
  • 1D / norm / bias / running stats / scalar metadata → unchanged dtype
  • 3D (depth-wise conv kernels) → unchanged F16 (small + ragged)

Q8_0 cuts ~50% off F16 size with quality at "indistinguishable from F16" for
inference (8-bit per-block scale, group-32). gguf-py's quants.quantize() ships
real Q8_0 (not stub) — verified before run.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

from gguf import GGMLQuantizationType, quants


KEEP_AS_IS_NAME_HINTS = (
    "norm", "bias", "running_mean", "running_var",
    "pos_bias_u", "pos_bias_v",  # tiny rel-pos bias tensors (8, 128)
    "preprocessor.fb", "preprocessor.window",  # mel filterbank + window — F32
)


def pick_qtype(name: str, gguf_shape: tuple, src_qtype: GGMLQuantizationType) -> GGMLQuantizationType:
    """gguf_shape is GGUF-storage order (innermost first). For Q8_0 we need the
    INNERMOST (contiguous, gguf_shape[0]) dim divisible by 32."""
    if any(h in name for h in KEEP_AS_IS_NAME_HINTS):
        return src_qtype
    if len(gguf_shape) <= 1 or len(gguf_shape) >= 3:
        return src_qtype
    inner, outer = gguf_shape[0], gguf_shape[1]
    if inner % 32 == 0 and outer >= 32:
        return GGMLQuantizationType.Q8_0
    return src_qtype


def quantized_nbytes(shape, qtype):
    if qtype == GGMLQuantizationType.F32:
        return int(np.prod(shape)) * 4
    if qtype == GGMLQuantizationType.F16:
        return int(np.prod(shape)) * 2
    byte_shape = quants.quant_shape_to_byte_shape(shape, qtype)
    return int(np.prod(byte_shape))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="F16 GGUF input")
    p.add_argument("--output", required=True, type=Path, help="Q8_0 GGUF output")
    args = p.parse_args()

    print(f"reading {args.input}")
    r = gguf.GGUFReader(str(args.input))

    # Carry over the original arch + KV table.
    arch_field = r.get_field("general.architecture")
    arch = "".join(chr(c) for c in arch_field.parts[arch_field.data[0]]) if arch_field else "parakeet"
    print(f"  arch: {arch}")
    print(f"  tensors: {len(r.tensors)}")

    # Plan first so we can print summary.
    plan = []
    for t in r.tensors:
        src_qt = t.tensor_type
        new_qt = pick_qtype(t.name, tuple(t.shape), src_qt)
        plan.append((t, src_qt, new_qt))

    n_q8 = sum(1 for _, _, q in plan if q == GGMLQuantizationType.Q8_0)
    n_f16 = sum(1 for _, _, q in plan if q == GGMLQuantizationType.F16)
    n_f32 = sum(1 for _, _, q in plan if q == GGMLQuantizationType.F32)
    other = len(plan) - n_q8 - n_f16 - n_f32
    print(f"  plan: {n_q8} Q8_0 + {n_f16} F16 + {n_f32} F32 + {other} other")

    writer = gguf.GGUFWriter(str(args.output), arch, use_temp_file=False)

    # Replay every KV that isn't a tensor info.
    SKIP_KV_PREFIXES = ("GGUF.",)  # gguf reserved
    SKIP_KV_NAMES = {"general.architecture"}  # set implicitly by GGUFWriter(arch=...)
    for field in r.fields.values():
        # GGUFReader exposes both KV and tensor INFO under .fields. Skip
        # anything that looks like a tensor info (it'll be re-added by add_tensor).
        if field.name in {t.name for t in r.tensors}:
            continue
        if any(field.name.startswith(p) for p in SKIP_KV_PREFIXES):
            continue
        if field.name in SKIP_KV_NAMES:
            continue
        # Re-emit the field. GGUFReader gives us raw bytes / parts.
        # Easier: rely on add_array / add_uint32 / etc based on type.
        try:
            t = field.types[0]
            if t == gguf.GGUFValueType.STRING:
                s = bytes(field.parts[field.data[0]]).decode("utf-8")
                writer.add_string(field.name, s)
            elif t == gguf.GGUFValueType.UINT32:
                writer.add_uint32(field.name, int(field.parts[field.data[0]][0]))
            elif t == gguf.GGUFValueType.INT32:
                writer.add_int32(field.name, int(field.parts[field.data[0]][0]))
            elif t == gguf.GGUFValueType.FLOAT32:
                writer.add_float32(field.name, float(field.parts[field.data[0]][0]))
            elif t == gguf.GGUFValueType.BOOL:
                writer.add_bool(field.name, bool(field.parts[field.data[0]][0]))
            elif t == gguf.GGUFValueType.ARRAY:
                # Element type is field.types[1]
                et = field.types[1]
                vals = []
                for idx in field.data:
                    part = field.parts[idx]
                    if et == gguf.GGUFValueType.STRING:
                        vals.append(bytes(part).decode("utf-8"))
                    elif hasattr(part, "tolist"):
                        v = part.tolist()
                        # Some scalar arrays come back as nested [v]
                        if isinstance(v, list) and len(v) == 1:
                            vals.append(v[0])
                        else:
                            vals.append(v)
                    else:
                        vals.append(part)
                writer.add_array(field.name, vals)
            else:
                print(f"  WARN: skipping KV {field.name} (type {t})")
        except Exception as e:
            print(f"  WARN: KV {field.name} replay failed: {e}")

    # Now add tensors with their (possibly new) quant type.
    print("quantizing tensors...")
    for i, (t, src_qt, new_qt) in enumerate(plan):
        gguf_shape = tuple(t.shape)
        # GGUFReader returns shape in GGUF order (innermost first). To
        # interpret the raw bytes as a numpy array we need numpy order
        # (outermost first), which is the reverse.
        np_shape = tuple(reversed(gguf_shape))
        if src_qt == GGMLQuantizationType.F32:
            arr = np.array(t.data, copy=False).astype(np.float32).reshape(np_shape)
        elif src_qt == GGMLQuantizationType.F16:
            arr = np.array(t.data, copy=False).view(np.float16).reshape(np_shape).astype(np.float32)
        else:
            print(f"  passthrough {t.name} ({src_qt.name})")
            writer.add_tensor(t.name, np.array(t.data, copy=True), raw_dtype=src_qt)
            continue

        if new_qt == GGMLQuantizationType.Q8_0:
            qbytes = quants.quantize(arr, GGMLQuantizationType.Q8_0)
            writer.add_tensor(t.name, qbytes, raw_dtype=GGMLQuantizationType.Q8_0)
        elif new_qt == GGMLQuantizationType.F16:
            writer.add_tensor(t.name, arr.astype(np.float16))
        else:  # F32
            writer.add_tensor(t.name, arr.astype(np.float32))

        if (i + 1) % 100 == 0 or i == len(plan) - 1:
            print(f"  {i+1}/{len(plan)}  {t.name}  {src_qt.name}->{new_qt.name}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = args.output.stat().st_size
    in_size = args.input.stat().st_size
    print(f"\ndone: {args.output}  {out_size/1e6:.1f} MB  ({100*out_size/in_size:.1f}% of input)")


if __name__ == "__main__":
    main()
