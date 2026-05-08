// parakeet.cpp megakernel-v0 (local fork only) — multi-block fused LSTM
// step + joint head.
//
// Lap-7 design lessons (single-block kernel was a dead end):
//   - Single-block "boring" kernel runs on 1 SM. RTX 3060 has 28 SMs.
//     For our shape (LSTM gates: 4H × H GEMV at H=640) the per-SM
//     compute ceiling (~228 GFLOPS) makes the kernel ~50-100 µs even
//     with coalesced loads — slower than lap-6's distributed multi-op
//     path.
//   - The lap-6 floor (pred p50 = 69 µs) decomposes as ~16 × 3-4 µs of
//     CPU-side ggml-cuda per-NODE dispatch glue + ~15 µs of base call
//     overhead + sub-µs/op multi-SM GPU compute. The lever is per-node
//     glue, NOT compute speed.
//
// Multi-block fused: 1 ggml node pays per-node glue once; inside the
// dispatch we issue several cudaLaunchKernel calls that the CUDA graph
// captures and replays as one cudaGraphLaunch. Each launch is a small
// multi-block GEMV that uses several SMs concurrently — same compute
// shape as ggml-cuda's mul_mat at M=1.
//
// LSTM step: 4 internal kernels per ggml node
//   1. gates layer 0  — multi-block GEMV, 4H gates from embed[tok_id] + h0
//   2. activate layer 0 — single block, H threads; updates c0/h0 in place
//   3. gates layer 1  — multi-block GEMV, 4H gates from new h0 + h1
//   4. activate layer 1 — single block, H threads; updates c1/h1 in place
//
// Joint head: 2 internal kernels per ggml node
//   1. stage 1 mid    — multi-block GEMV with ReLU, J outputs from enc_t + h1
//   2. stage 2 logits — multi-block GEMV, V outputs from mid_buf
//
// Scratch buffers (gates_buf [4H], mid_buf [J]) live in tdt_gpu state_buf
// alongside h0/c0/h1/c1 and are passed as src tensors so gallocr leaves
// them alone (already-allocated persistent backend buffers).

#include "parakeet_megakernel.cuh"

#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  1);
    return v;
}

// ---------------------------------------------------------------------------
// LSTM step — 4 kernels
// ---------------------------------------------------------------------------

// Layer 0 gates: gates[g] = b_ih[g] + b_hh[g] + W_ih[g,:] . x + W_hh[g,:] . h0
//   x = embed_w[tok_id, :] (F16 → F32 inline)
//
// Block-row layout: each block computes 4H/gridDim.x output gates. Within a
// block, 8 warps × 1 gate-per-warp → coalesced GEMV per warp. Smem holds
// the H-element x vector (loaded once per block, ~1.3 KB at H=640).
__global__ void parakeet_lstm_gates_l0_kernel(
        const __half * __restrict__ embed_w,
        const __half * __restrict__ w_ih, const float * __restrict__ b_ih,
        const __half * __restrict__ w_hh, const float * __restrict__ b_hh,
        const float  * __restrict__ h0,
        const int    * __restrict__ tok_id_ptr,
        float        * __restrict__ gates_out,
        int H) {
    extern __shared__ float smem[];
    float * sh_x = smem;   // [H]

    const int tid     = threadIdx.x;
    const int bdim    = blockDim.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int n_warps = bdim >> 5;

    // Each block does its own embed lookup into smem (cheap; ~1.3 KB).
    const int tok_id = tok_id_ptr[0];
    const __half * row = embed_w + (size_t)tok_id * H;
    for (int i = tid; i < H; i += bdim) {
        sh_x[i] = __half2float(row[i]);
    }
    __syncthreads();

    // Block g-range
    const int n_g          = 4 * H;
    const int g_per_block  = (n_g + gridDim.x - 1) / gridDim.x;
    const int g_block_base = blockIdx.x * g_per_block;
    const int g_block_end  = (g_block_base + g_per_block < n_g) ? (g_block_base + g_per_block) : n_g;

    for (int g = g_block_base + warp_id; g < g_block_end; g += n_warps) {
        const __half * w_ih_row = w_ih + (size_t)g * H;
        const __half * w_hh_row = w_hh + (size_t)g * H;
        float partial = 0.0f;
        for (int k = lane_id; k < H; k += 32) {
            partial += __half2float(w_ih_row[k]) * sh_x[k];
            partial += __half2float(w_hh_row[k]) * h0[k];
        }
        const float reduced = warp_sum(partial);
        if (lane_id == 0) {
            gates_out[g] = reduced + b_ih[g] + b_hh[g];
        }
    }
}

// Layer 1 gates: same shape as layer 0 but reads new h0 (already updated by
// activate_l0) as input, h1 as recurrent state. No embed lookup.
__global__ void parakeet_lstm_gates_l1_kernel(
        const __half * __restrict__ w_ih, const float * __restrict__ b_ih,
        const __half * __restrict__ w_hh, const float * __restrict__ b_hh,
        const float  * __restrict__ h0_new,   // input x (= post-activate h0)
        const float  * __restrict__ h1,
        float        * __restrict__ gates_out,
        int H) {
    extern __shared__ float smem[];
    float * sh_x = smem;   // [H]

    const int tid     = threadIdx.x;
    const int bdim    = blockDim.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int n_warps = bdim >> 5;

    // Stage h0_new into smem so subsequent rows reuse it from L1 cache.
    for (int i = tid; i < H; i += bdim) {
        sh_x[i] = h0_new[i];
    }
    __syncthreads();

    const int n_g          = 4 * H;
    const int g_per_block  = (n_g + gridDim.x - 1) / gridDim.x;
    const int g_block_base = blockIdx.x * g_per_block;
    const int g_block_end  = (g_block_base + g_per_block < n_g) ? (g_block_base + g_per_block) : n_g;

    for (int g = g_block_base + warp_id; g < g_block_end; g += n_warps) {
        const __half * w_ih_row = w_ih + (size_t)g * H;
        const __half * w_hh_row = w_hh + (size_t)g * H;
        float partial = 0.0f;
        for (int k = lane_id; k < H; k += 32) {
            partial += __half2float(w_ih_row[k]) * sh_x[k];
            partial += __half2float(w_hh_row[k]) * h1[k];
        }
        const float reduced = warp_sum(partial);
        if (lane_id == 0) {
            gates_out[g] = reduced + b_ih[g] + b_hh[g];
        }
    }
}

// Activate kernel: read gates [4H], update c/h state in place.
//   c_new = sigmoid(g_f) * c_old + sigmoid(g_i) * tanh(g_g)
//   h_new = sigmoid(g_o) * tanh(c_new)
//
// Single block, H threads (with H ≤ 1024). One thread per state element.
__global__ void parakeet_lstm_activate_kernel(
        const float * __restrict__ gates,
        float       * __restrict__ c_state,
        float       * __restrict__ h_state,
        int H) {
    const int i = threadIdx.x;
    if (i >= H) return;

    const float ig = 1.0f / (1.0f + expf(-gates[0 * H + i]));
    const float fg = 1.0f / (1.0f + expf(-gates[1 * H + i]));
    const float gg = tanhf(gates[2 * H + i]);
    const float og = 1.0f / (1.0f + expf(-gates[3 * H + i]));
    const float c_old = c_state[i];
    const float c_new = fg * c_old + ig * gg;
    const float h_new = og * tanhf(c_new);
    c_state[i] = c_new;
    h_state[i] = h_new;
}

// ---------------------------------------------------------------------------
// Joint head — 2 kernels
// ---------------------------------------------------------------------------

// Stage 1: mid[j] = ReLU(enc_w[j,:].enc_t + enc_b[j] + pred_w[j,:].h_1 + pred_b[j])
//   enc_w ne=[D, J]; pred_w ne=[P, J].
//
// Multi-block GEMV: each block handles J/gridDim.x outputs. Smem stages
// enc_t and h_1 for warp-coalesced reads.
__global__ void parakeet_joint_stage1_kernel(
        const __half * __restrict__ enc_w,  const float * __restrict__ enc_b,
        const __half * __restrict__ pred_w, const float * __restrict__ pred_b,
        const float  * __restrict__ enc_t,  // [D]
        const float  * __restrict__ h_1,    // [P]
        float        * __restrict__ mid_out,
        int J, int D, int P) {
    extern __shared__ float smem[];
    float * sh_enc = smem;            // [D]
    float * sh_h   = sh_enc + D;      // [P]

    const int tid     = threadIdx.x;
    const int bdim    = blockDim.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int n_warps = bdim >> 5;

    for (int i = tid; i < D; i += bdim) sh_enc[i] = enc_t[i];
    for (int i = tid; i < P; i += bdim) sh_h[i]   = h_1[i];
    __syncthreads();

    const int j_per_block  = (J + gridDim.x - 1) / gridDim.x;
    const int j_block_base = blockIdx.x * j_per_block;
    const int j_block_end  = (j_block_base + j_per_block < J) ? (j_block_base + j_per_block) : J;

    for (int j = j_block_base + warp_id; j < j_block_end; j += n_warps) {
        const __half * enc_w_row  = enc_w  + (size_t)j * D;
        const __half * pred_w_row = pred_w + (size_t)j * P;
        float partial = 0.0f;
        for (int k = lane_id; k < D; k += 32) partial += __half2float(enc_w_row[k])  * sh_enc[k];
        for (int k = lane_id; k < P; k += 32) partial += __half2float(pred_w_row[k]) * sh_h[k];
        const float reduced = warp_sum(partial);
        if (lane_id == 0) {
            const float v = reduced + enc_b[j] + pred_b[j];
            mid_out[j] = fmaxf(0.0f, v);
        }
    }
}

// Stage 2: logits[v] = out_w[v,:] . mid + out_b[v]   (out_w ne=[J, V])
__global__ void parakeet_joint_stage2_kernel(
        const __half * __restrict__ out_w, const float * __restrict__ out_b,
        const float  * __restrict__ mid_buf,    // [J]
        float        * __restrict__ logits,     // [V]
        int V, int J) {
    extern __shared__ float smem[];
    float * sh_mid = smem;   // [J]

    const int tid     = threadIdx.x;
    const int bdim    = blockDim.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int n_warps = bdim >> 5;

    for (int i = tid; i < J; i += bdim) sh_mid[i] = mid_buf[i];
    __syncthreads();

    const int v_per_block  = (V + gridDim.x - 1) / gridDim.x;
    const int v_block_base = blockIdx.x * v_per_block;
    const int v_block_end  = (v_block_base + v_per_block < V) ? (v_block_base + v_per_block) : V;

    for (int v = v_block_base + warp_id; v < v_block_end; v += n_warps) {
        const __half * out_w_row = out_w + (size_t)v * J;
        float partial = 0.0f;
        for (int k = lane_id; k < J; k += 32) partial += __half2float(out_w_row[k]) * sh_mid[k];
        const float reduced = warp_sum(partial);
        if (lane_id == 0) {
            logits[v] = reduced + out_b[v];
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch wrappers
// ---------------------------------------------------------------------------

namespace {

// First-call wall-clock dump for the entire ggml dispatch (all internal
// kernels) when PARAKEET_MEGAKERNEL_TIMING=1. Implementation note: the
// captured CUDA graph means kernel-level event timing during normal
// operation is meaningless (events get captured into the graph). To get a
// representative number, run with GGML_CUDA_DISABLE_GRAPHS=1 alongside.
template <typename Launch>
inline void run_with_optional_timing(cudaStream_t stream, const char * name,
                                     bool & fired, Launch launch) {
    const char * env = std::getenv("PARAKEET_MEGAKERNEL_TIMING");
    const bool enabled = env && env[0] && env[0] != '0';
    if (!enabled || fired) {
        launch(stream);
        return;
    }
    fired = true;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, stream);
    launch(stream);
    cudaEventRecord(b, stream);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    std::fprintf(stderr, "parakeet[mega] %s = %.1f us\n", name, ms * 1000.0f);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

// Block-grid sizing for multi-block GEMV. Tuned for RTX 3060 (28 SMs):
// 8 blocks splits LSTM gates into 320 outputs/block (manageable) and 32
// blocks splits the V=8198 logits at ~256 outputs/block. For smaller
// stages (J=640) we use 8 blocks at ~80 outputs/block.
constexpr int LSTM_GATES_BLOCKS  = 8;
constexpr int JOINT_STAGE1_BLOCKS = 8;
constexpr int JOINT_STAGE2_BLOCKS = 32;
constexpr int BLOCK_THREADS       = 256;

}  // namespace

void ggml_cuda_op_parakeet_lstm_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_PARAKEET_LSTM_STEP);

    const ggml_tensor * embed_w    = dst->src[ 0];
    const ggml_tensor * lstm0_w_ih = dst->src[ 1];
    const ggml_tensor * lstm0_b_ih = dst->src[ 2];
    const ggml_tensor * lstm0_w_hh = dst->src[ 3];
    const ggml_tensor * lstm0_b_hh = dst->src[ 4];
    const ggml_tensor * lstm1_w_ih = dst->src[ 5];
    const ggml_tensor * lstm1_b_ih = dst->src[ 6];
    const ggml_tensor * lstm1_w_hh = dst->src[ 7];
    const ggml_tensor * lstm1_b_hh = dst->src[ 8];
    ggml_tensor *       h0         = dst->src[ 9];
    ggml_tensor *       c0         = dst->src[10];
    ggml_tensor *       h1         = dst->src[11];
    ggml_tensor *       c1         = dst->src[12];
    const ggml_tensor * tok_id     = dst->src[13];
    ggml_tensor *       gates_buf  = dst->src[14];

    const int H = (int) h0->ne[0];
    GGML_ASSERT(H > 0 && H <= 1024 && "activate kernel uses 1 thread per state element");
    GGML_ASSERT(gates_buf->ne[0] == 4 * H);

    cudaStream_t stream = ctx.stream();

    auto launch_all = [&](cudaStream_t s) {
        const size_t smem_x = (size_t)H * sizeof(float);

        // 1. Layer 0 gates → gates_buf
        parakeet_lstm_gates_l0_kernel<<<LSTM_GATES_BLOCKS, BLOCK_THREADS, smem_x, s>>>(
            (const __half *) embed_w->data,
            (const __half *) lstm0_w_ih->data, (const float *) lstm0_b_ih->data,
            (const __half *) lstm0_w_hh->data, (const float *) lstm0_b_hh->data,
            (const float *)  h0->data,
            (const int *)    tok_id->data,
            (float *)        gates_buf->data,
            H);

        // 2. Layer 0 activate → updates c0, h0 in place
        parakeet_lstm_activate_kernel<<<1, H, 0, s>>>(
            (const float *) gates_buf->data,
            (float *) c0->data, (float *) h0->data,
            H);

        // 3. Layer 1 gates: input is the new h0 (just written by step 2),
        //    recurrent state is h1. Reuses gates_buf.
        parakeet_lstm_gates_l1_kernel<<<LSTM_GATES_BLOCKS, BLOCK_THREADS, smem_x, s>>>(
            (const __half *) lstm1_w_ih->data, (const float *) lstm1_b_ih->data,
            (const __half *) lstm1_w_hh->data, (const float *) lstm1_b_hh->data,
            (const float *)  h0->data,
            (const float *)  h1->data,
            (float *)        gates_buf->data,
            H);

        // 4. Layer 1 activate → updates c1, h1 in place
        parakeet_lstm_activate_kernel<<<1, H, 0, s>>>(
            (const float *) gates_buf->data,
            (float *) c1->data, (float *) h1->data,
            H);
    };

    static bool s_fired_lstm = false;
    run_with_optional_timing(stream, "lstm_step (4 kernels)", s_fired_lstm, launch_all);
}

void ggml_cuda_op_parakeet_joint(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_PARAKEET_JOINT);

    const ggml_tensor * enc_w   = dst->src[0];
    const ggml_tensor * enc_b   = dst->src[1];
    const ggml_tensor * pred_w  = dst->src[2];
    const ggml_tensor * pred_b  = dst->src[3];
    const ggml_tensor * out_w   = dst->src[4];
    const ggml_tensor * out_b   = dst->src[5];
    const ggml_tensor * enc_t   = dst->src[6];
    const ggml_tensor * h_1     = dst->src[7];
    ggml_tensor *       mid_buf = dst->src[8];

    const int D = (int) enc_w->ne[0];
    const int J = (int) enc_w->ne[1];
    const int P = (int) pred_w->ne[0];
    const int V = (int) out_w->ne[1];
    GGML_ASSERT(mid_buf->ne[0] == J);
    GGML_ASSERT(dst->ne[0] == V);

    cudaStream_t stream = ctx.stream();

    auto launch_all = [&](cudaStream_t s) {
        const size_t smem_s1 = (size_t)(D + P) * sizeof(float);
        parakeet_joint_stage1_kernel<<<JOINT_STAGE1_BLOCKS, BLOCK_THREADS, smem_s1, s>>>(
            (const __half *) enc_w->data,  (const float *) enc_b->data,
            (const __half *) pred_w->data, (const float *) pred_b->data,
            (const float *)  enc_t->data,
            (const float *)  h_1->data,
            (float *)        mid_buf->data,
            J, D, P);

        const size_t smem_s2 = (size_t)J * sizeof(float);
        parakeet_joint_stage2_kernel<<<JOINT_STAGE2_BLOCKS, BLOCK_THREADS, smem_s2, s>>>(
            (const __half *) out_w->data, (const float *) out_b->data,
            (const float *)  mid_buf->data,
            (float *)        dst->data,
            V, J);
    };

    static bool s_fired_joint = false;
    run_with_optional_timing(stream, "joint (2 kernels)", s_fired_joint, launch_all);
}
