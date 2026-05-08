#!/usr/bin/env python3
"""Re-pack a parakeet F16 GGUF to Q4_K (pure, not Q4_K_M mix) by calling the
canonical upstream `ggml_quantize_chunk` from libggml.so via ctypes.

Why this and not llama-quantize: llama-quantize validates input as a known
LLM architecture (llama/gpt2/falcon/etc.) and refuses parakeet GGUFs even
with --override-kv. Why this and not crispasr-quantize: that's a thin C++
wrapper around the same `ggml_quantize_chunk`; ctypes-calling the symbol
directly skips both wrappers entirely.

Tensor strategy mirrors parakeet-f16-to-q8_0.py: 2D weights with last dim
divisible by 256 → Q4_K (Q4_K's super-block is 256 elements). Weights with
last dim only divisible by 32 (not 256) → fall back to Q8_0. Everything
else (norms, biases, 3D conv kernels, mel filterbank) → unchanged.
"""
import argparse
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

from gguf import GGMLQuantizationType, quants


# ggml_type enum (from ggml.h)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14


def load_ggml():
    # ggml-base + ggml-cpu must be loaded with RTLD_GLOBAL so libggml.so's
    # backend registration finds them. Order: base first.
    ctypes.CDLL("libggml-base.so.0", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("libggml-cpu.so.0", mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL("libggml.so.0", mode=ctypes.RTLD_GLOBAL)

    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,                              # ggml_type
        ctypes.POINTER(ctypes.c_float),            # const float* src
        ctypes.c_void_p,                           # void* dst
        ctypes.c_int64,                            # start
        ctypes.c_int64,                            # nrows
        ctypes.c_int64,                            # n_per_row
        ctypes.POINTER(ctypes.c_float),            # const float* imatrix (nullable)
    ]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t

    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_row_size.restype = ctypes.c_size_t

    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    lib.ggml_quantize_init.restype = None

    lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
    lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool

    return lib


def quantize_via_ggml(lib, arr_f32: np.ndarray, ggml_type: int) -> np.ndarray:
    """Returns uint8 byte buffer of quantized data (1D, contains nrows * row_bytes)."""
    assert arr_f32.dtype == np.float32 and arr_f32.ndim == 2 and arr_f32.flags["C_CONTIGUOUS"]
    nrows, n_per_row = arr_f32.shape
    row_bytes = lib.ggml_row_size(ggml_type, n_per_row)
    total_bytes = row_bytes * nrows
    out = np.zeros(total_bytes, dtype=np.uint8)
    written = lib.ggml_quantize_chunk(
        ggml_type,
        arr_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.c_void_p),
        0, nrows, n_per_row,
        None,  # imatrix=null → fall back to no-imatrix path
    )
    if written != total_bytes:
        raise RuntimeError(f"ggml_quantize_chunk wrote {written}, expected {total_bytes}")
    return out


KEEP_AS_IS_NAME_HINTS = (
    "norm", "bias", "running_mean", "running_var",
    "pos_bias_u", "pos_bias_v",
    "preprocessor.fb", "preprocessor.window",
    # Output-sensitive — keep at F16. Quantizing the predictor embedding
    # or joint output projection breaks token argmax decisions.
    "decoder.embed.weight",
    "joint.out.weight",
)

# Map gguf-py quant type → ggml int enum
GGUF_TO_GGML = {
    GGMLQuantizationType.F32: GGML_TYPE_F32,
    GGMLQuantizationType.F16: GGML_TYPE_F16,
    GGMLQuantizationType.Q8_0: GGML_TYPE_Q8_0,
    GGMLQuantizationType.Q4_K: GGML_TYPE_Q4_K,
}


def pick_qtype(name: str, gguf_shape: tuple, src_qtype: GGMLQuantizationType) -> GGMLQuantizationType:
    """gguf_shape is GGUF-storage order (innermost first). For Q4_K we need
    INNERMOST (gguf_shape[0]) divisible by 256; Q8_0 falls back at 32."""
    if any(h in name for h in KEEP_AS_IS_NAME_HINTS):
        return src_qtype
    if len(gguf_shape) <= 1 or len(gguf_shape) >= 3:
        return src_qtype
    inner, outer = gguf_shape[0], gguf_shape[1]
    if inner % 256 == 0 and outer >= 32:
        return GGMLQuantizationType.Q4_K
    if inner % 32 == 0 and outer >= 32:
        return GGMLQuantizationType.Q8_0
    return src_qtype


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    lib = load_ggml()
    lib.ggml_quantize_init(GGML_TYPE_Q4_K)
    lib.ggml_quantize_init(GGML_TYPE_Q8_0)

    print(f"reading {args.input}")
    r = gguf.GGUFReader(str(args.input))

    arch_field = r.get_field("general.architecture")
    arch = "".join(chr(c) for c in arch_field.parts[arch_field.data[0]]) if arch_field else "parakeet"
    print(f"  arch: {arch}, tensors: {len(r.tensors)}")

    plan = []
    for t in r.tensors:
        new_qt = pick_qtype(t.name, tuple(t.shape), t.tensor_type)
        plan.append((t, t.tensor_type, new_qt))

    counts = {}
    for _, _, q in plan:
        counts[q.name] = counts.get(q.name, 0) + 1
    print(f"  plan: {counts}")

    writer = gguf.GGUFWriter(str(args.output), arch, use_temp_file=False)

    SKIP_KV_PREFIXES = ("GGUF.",)
    SKIP_KV_NAMES = {"general.architecture"}
    for field in r.fields.values():
        if field.name in {t.name for t in r.tensors}:
            continue
        if any(field.name.startswith(p) for p in SKIP_KV_PREFIXES):
            continue
        if field.name in SKIP_KV_NAMES:
            continue
        try:
            t0 = field.types[0]
            if t0 == gguf.GGUFValueType.STRING:
                writer.add_string(field.name, bytes(field.parts[field.data[0]]).decode("utf-8"))
            elif t0 == gguf.GGUFValueType.UINT32:
                writer.add_uint32(field.name, int(field.parts[field.data[0]][0]))
            elif t0 == gguf.GGUFValueType.INT32:
                writer.add_int32(field.name, int(field.parts[field.data[0]][0]))
            elif t0 == gguf.GGUFValueType.FLOAT32:
                writer.add_float32(field.name, float(field.parts[field.data[0]][0]))
            elif t0 == gguf.GGUFValueType.BOOL:
                writer.add_bool(field.name, bool(field.parts[field.data[0]][0]))
            elif t0 == gguf.GGUFValueType.ARRAY:
                et = field.types[1]
                vals = []
                for idx in field.data:
                    part = field.parts[idx]
                    if et == gguf.GGUFValueType.STRING:
                        vals.append(bytes(part).decode("utf-8"))
                    elif hasattr(part, "tolist"):
                        v = part.tolist()
                        vals.append(v[0] if isinstance(v, list) and len(v) == 1 else v)
                    else:
                        vals.append(part)
                writer.add_array(field.name, vals)
        except Exception as e:
            print(f"  KV {field.name} skipped: {e}")

    print("quantizing tensors...")
    for i, (t, src_qt, new_qt) in enumerate(plan):
        gguf_shape = tuple(t.shape)
        # GGUF order is innermost-first; numpy is outermost-first → reverse.
        np_shape = tuple(reversed(gguf_shape))
        if src_qt == GGMLQuantizationType.F32:
            arr32 = np.array(t.data, copy=False).astype(np.float32).reshape(np_shape)
        elif src_qt == GGMLQuantizationType.F16:
            arr32 = np.array(t.data, copy=False).view(np.float16).reshape(np_shape).astype(np.float32)
        else:
            print(f"  passthrough quantized {t.name} ({src_qt.name})")
            writer.add_tensor(t.name, np.array(t.data, copy=True), raw_dtype=src_qt)
            continue

        if new_qt in (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q8_0):
            arr32_c = np.ascontiguousarray(arr32)
            # ggml_quantize_chunk wants nrows × n_per_row in numpy order
            # (n_per_row = innermost = arr.shape[-1] = gguf_shape[0]).
            qbytes_flat = quantize_via_ggml(lib, arr32_c, GGUF_TO_GGML[new_qt])
            row_bytes = lib.ggml_row_size(GGUF_TO_GGML[new_qt], arr32_c.shape[-1])
            qbytes = qbytes_flat.reshape(arr32_c.shape[0], row_bytes)
            writer.add_tensor(t.name, qbytes, raw_dtype=new_qt)
        elif new_qt == GGMLQuantizationType.F16:
            writer.add_tensor(t.name, arr32.astype(np.float16))
        else:
            writer.add_tensor(t.name, arr32.astype(np.float32))

        if (i + 1) % 100 == 0 or i == len(plan) - 1:
            print(f"  {i+1}/{len(plan)}  {t.name}  {src_qt.name}->{new_qt.name}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = args.output.stat().st_size
    in_size = args.input.stat().st_size
    print(f"\ndone: {args.output}  {out_size/1e6:.1f} MB  ({100*out_size/in_size:.1f}% of F16)")


if __name__ == "__main__":
    main()
