// parakeet.cpp megakernel-v0 (local fork only).
//
// CUDA-side dispatch entry points for the two fused decode-side ops:
//   GGML_OP_PARAKEET_LSTM_STEP — predictor 2-layer LSTM step (16 ops -> 1)
//   GGML_OP_PARAKEET_JOINT     — joint head (enc/pred proj -> ReLU -> out)  (8 ops -> 1)
//
// Each entry point is invoked from ggml_cuda_compute_forward and runs one
// __global__ launch on the cuda_ctx stream. Both kernels are single-block
// (one SM, 256 threads), latency-tuned for the RTX 3060 / sm_86 budget
// targeted in MEGAKERNEL-SPEC.md. Per-call kernel-level wall-clock is
// measured once via cudaEventRecord when PARAKEET_MEGAKERNEL_TIMING=1.
//
// Type policy (Commit 2/3): F16 weights only. The graph builder in
// src/parakeet.cpp must guarantee that all weight sources are F16 before
// emitting these ops. The dispatch asserts on type mismatch — that is a
// programmer error, not a runtime fallback.

#pragma once

#include "common.cuh"

void ggml_cuda_op_parakeet_lstm_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_parakeet_joint    (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
