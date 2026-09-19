#if defined(__HIP_PLATFORM_AMD__) && defined(UMA_QMOE_USE_WMMA)
#if defined(__HIP_NO_HALF_OPERATORS__)
#undef __HIP_NO_HALF_OPERATORS__
#endif
#if defined(__HIP_NO_HALF_CONVERSIONS__)
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#include <rocwmma/rocwmma.hpp>
#endif

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kHiddenSize = 2048;
constexpr int kIntermediateSize = 1024;
constexpr int kExpertCount = 64;
constexpr int kTopK = 8;
constexpr int kRouteTile = 16;
constexpr int kOutputTile = 16;
constexpr int kReductionTile = 64;
constexpr int kVectorOutputTile = 32;
constexpr int kOutputsPerThread = kVectorOutputTile / kOutputTile;

// Map a compact route-tile id to one expert-local tile.  The host launches an
// upper bound of ceil(total_routes / tile) + experts - 1 blocks; surplus blocks
// exit.  This avoids the old experts * ceil(tokens / tile) rectangular grid,
// whose empty blocks dominated the frozen 128-token RouteTrace.
__device__ __forceinline__ bool resolve_compact_route_tile(
    const std::int64_t* __restrict__ expert_offsets,
    std::int64_t compact_tile,
    std::int64_t* expert,
    std::int64_t* route_base) {
  std::int64_t tile_offset = 0;
#pragma unroll
  for (std::int64_t candidate = 0; candidate < kExpertCount; ++candidate) {
    const std::int64_t count =
        expert_offsets[candidate + 1] - expert_offsets[candidate];
    const std::int64_t tile_count =
        (count + kRouteTile - 1) / kRouteTile;
    if (compact_tile < tile_offset + tile_count) {
      *expert = candidate;
      *route_base = expert_offsets[candidate] +
                    (compact_tile - tile_offset) * kRouteTile;
      return true;
    }
    tile_offset += tile_count;
  }
  *expert = -1;
  *route_base = -1;
  return false;
}

__device__ __forceinline__ int unpack_low(const std::uint8_t value) {
  const int nibble = value & 0x0f;
  return nibble >= 8 ? nibble - 16 : nibble;
}

__device__ __forceinline__ int unpack_high(const std::uint8_t value) {
  const int nibble = value >> 4;
  return nibble >= 8 ? nibble - 16 : nibble;
}

__global__ void packed_q4_bf16_gemv_kernel(
    const c10::BFloat16* __restrict__ input,
    const std::uint8_t* __restrict__ packed,
    const float* __restrict__ scales,
    c10::BFloat16* __restrict__ output,
    std::int64_t rows,
    std::int64_t output_features,
    std::int64_t input_features,
    std::int64_t group_size) {
  const std::int64_t output_feature = blockIdx.x;
  const std::int64_t row = blockIdx.y;
  if (row >= rows || output_feature >= output_features) return;

  float partial = 0.0f;
  const std::int64_t weight_row = output_feature * input_features;
  const std::int64_t input_row = row * input_features;
  for (std::int64_t byte_column = threadIdx.x;
       byte_column < input_features / 2; byte_column += blockDim.x) {
    const std::int64_t column = byte_column * 2;
    const std::int64_t weight_index = weight_row + column;
    const std::uint8_t byte = packed[weight_index >> 1];
    const float scale = scales[weight_index / group_size];
    partial += static_cast<float>(input[input_row + column]) *
               (static_cast<float>(unpack_low(byte)) * scale);
    partial += static_cast<float>(input[input_row + column + 1]) *
               (static_cast<float>(unpack_high(byte)) * scale);
  }

  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row * output_features + output_feature] =
        static_cast<c10::BFloat16>(reduction[0]);
  }
}

// One block computes one routed expert's SwiGLU intermediate element. Gate
// and Up share activation reads and are reduced in the same launch.
__global__ void packed_q4_gate_up_swiglu_kernel(
    const c10::BFloat16* __restrict__ hidden,
    const std::int64_t* __restrict__ expert_indices,
    const std::int64_t* __restrict__ route_order,
    const std::uint8_t* __restrict__ gate_packed,
    const float* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_packed,
    const float* __restrict__ up_scales,
    c10::BFloat16* __restrict__ intermediate,
    std::int64_t rows,
    std::int64_t group_size) {
  const std::int64_t feature = blockIdx.x;
  const std::int64_t route_rank = blockIdx.y;
  const std::int64_t route_count = rows * kTopK;
  if (route_rank >= route_count || feature >= kIntermediateSize) return;
  const std::int64_t route =
      route_order == nullptr ? route_rank : route_order[route_rank];

  const std::int64_t token = route / kTopK;
  const std::int64_t expert = expert_indices[route];
  if (expert < 0 || expert >= kExpertCount) return;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kIntermediateSize) * kHiddenSize;
  constexpr std::int64_t kPackedStride = kMatrixElements / 2;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  const std::int64_t weight_row = feature * kHiddenSize;
  const std::int64_t packed_base = expert * kPackedStride + weight_row / 2;
  const std::int64_t scale_base =
      expert * scale_stride + weight_row / group_size;
  const std::int64_t input_base = token * kHiddenSize;

  float gate_partial = 0.0f;
  float up_partial = 0.0f;
  for (std::int64_t byte_column = threadIdx.x;
       byte_column < kHiddenSize / 2; byte_column += blockDim.x) {
    const std::int64_t column = byte_column * 2;
    const float input_low = static_cast<float>(hidden[input_base + column]);
    const float input_high =
        static_cast<float>(hidden[input_base + column + 1]);
    const float gate_scale = gate_scales[scale_base + column / group_size];
    const float up_scale = up_scales[scale_base + column / group_size];
    const std::uint8_t gate_byte = gate_packed[packed_base + byte_column];
    const std::uint8_t up_byte = up_packed[packed_base + byte_column];
    gate_partial += input_low * static_cast<float>(unpack_low(gate_byte)) *
                        gate_scale +
                    input_high * static_cast<float>(unpack_high(gate_byte)) *
                        gate_scale;
    up_partial += input_low * static_cast<float>(unpack_low(up_byte)) * up_scale +
                  input_high * static_cast<float>(unpack_high(up_byte)) * up_scale;
  }

  __shared__ float gate_reduction[kThreads];
  __shared__ float up_reduction[kThreads];
  gate_reduction[threadIdx.x] = gate_partial;
  up_reduction[threadIdx.x] = up_partial;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      gate_reduction[threadIdx.x] += gate_reduction[threadIdx.x + stride];
      up_reduction[threadIdx.x] += up_reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const c10::BFloat16 gate_bf16 =
        static_cast<c10::BFloat16>(gate_reduction[0]);
    const c10::BFloat16 up_bf16 =
        static_cast<c10::BFloat16>(up_reduction[0]);
    const float gate = static_cast<float>(gate_bf16);
    const float up = static_cast<float>(up_bf16);
    const float silu = gate / (1.0f + expf(-gate));
    intermediate[route * kIntermediateSize + feature] =
        static_cast<c10::BFloat16>(silu * up);
  }
}

// Prefill assigns one block to an expert/feature pair and walks that expert's
// stable route range.  The packed row stays hot while token activations vary.
__global__ void packed_q4_gate_up_grouped_kernel(
    const c10::BFloat16* __restrict__ hidden,
    const std::int64_t* __restrict__ route_order,
    const std::int64_t* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ gate_packed,
    const float* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_packed,
    const float* __restrict__ up_scales,
    c10::BFloat16* __restrict__ intermediate,
    std::int64_t group_size) {
  const std::int64_t feature = blockIdx.x;
  const std::int64_t expert = blockIdx.y;
  if (expert >= kExpertCount || feature >= kIntermediateSize) return;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kIntermediateSize) * kHiddenSize;
  constexpr std::int64_t kPackedStride = kMatrixElements / 2;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  const std::int64_t weight_row = feature * kHiddenSize;
  const std::int64_t packed_base = expert * kPackedStride + weight_row / 2;
  const std::int64_t scale_base =
      expert * scale_stride + weight_row / group_size;
  __shared__ float gate_reduction[kThreads];
  __shared__ float up_reduction[kThreads];

  for (std::int64_t rank = expert_offsets[expert];
       rank < expert_offsets[expert + 1]; ++rank) {
    const std::int64_t route = route_order[rank];
    const std::int64_t token = route / kTopK;
    const std::int64_t input_base = token * kHiddenSize;
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    for (std::int64_t byte_column = threadIdx.x;
         byte_column < kHiddenSize / 2; byte_column += blockDim.x) {
      const std::int64_t column = byte_column * 2;
      const float input_low = static_cast<float>(hidden[input_base + column]);
      const float input_high =
          static_cast<float>(hidden[input_base + column + 1]);
      const float gate_scale = gate_scales[scale_base + column / group_size];
      const float up_scale = up_scales[scale_base + column / group_size];
      const std::uint8_t gate_byte = gate_packed[packed_base + byte_column];
      const std::uint8_t up_byte = up_packed[packed_base + byte_column];
      gate_partial += input_low * static_cast<float>(unpack_low(gate_byte)) *
                          gate_scale +
                      input_high * static_cast<float>(unpack_high(gate_byte)) *
                          gate_scale;
      up_partial += input_low * static_cast<float>(unpack_low(up_byte)) *
                        up_scale +
                    input_high * static_cast<float>(unpack_high(up_byte)) *
                        up_scale;
    }
    gate_reduction[threadIdx.x] = gate_partial;
    up_reduction[threadIdx.x] = up_partial;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        gate_reduction[threadIdx.x] += gate_reduction[threadIdx.x + stride];
        up_reduction[threadIdx.x] += up_reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const c10::BFloat16 gate_bf16 =
          static_cast<c10::BFloat16>(gate_reduction[0]);
      const c10::BFloat16 up_bf16 =
          static_cast<c10::BFloat16>(up_reduction[0]);
      const float gate = static_cast<float>(gate_bf16);
      const float up = static_cast<float>(up_bf16);
      intermediate[route * kIntermediateSize + feature] =
          static_cast<c10::BFloat16>(gate / (1.0f + expf(-gate)) * up);
    }
    __syncthreads();
  }
}

// Prefill keeps routes grouped by expert through Down so consecutive blocks
// reuse the same packed matrix. Results are written at the original route
// index and reduced in original Top-K slot order by the next kernel.
__global__ void packed_q4_down_sorted_kernel(
    const c10::BFloat16* __restrict__ intermediate,
    const c10::BFloat16* __restrict__ routing_weights,
    const std::int64_t* __restrict__ route_order,
    const std::int64_t* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ down_packed,
    const float* __restrict__ down_scales,
    c10::BFloat16* __restrict__ route_output,
    std::int64_t group_size) {
  const std::int64_t output_feature = blockIdx.x;
  const std::int64_t expert = blockIdx.y;
  if (expert >= kExpertCount || output_feature >= kHiddenSize) return;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kHiddenSize) * kIntermediateSize;
  constexpr std::int64_t kPackedStride = kMatrixElements / 2;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  const std::int64_t weight_row = output_feature * kIntermediateSize;
  const std::int64_t packed_base = expert * kPackedStride + weight_row / 2;
  const std::int64_t scale_base =
      expert * scale_stride + weight_row / group_size;
  __shared__ float reduction[kThreads];
  for (std::int64_t rank = expert_offsets[expert];
       rank < expert_offsets[expert + 1]; ++rank) {
    const std::int64_t route = route_order[rank];
    const std::int64_t input_base = route * kIntermediateSize;
    float partial = 0.0f;
    for (std::int64_t byte_column = threadIdx.x;
         byte_column < kIntermediateSize / 2; byte_column += blockDim.x) {
      const std::int64_t column = byte_column * 2;
      const std::uint8_t byte = down_packed[packed_base + byte_column];
      const float scale = down_scales[scale_base + column / group_size];
      partial += static_cast<float>(intermediate[input_base + column]) *
                     static_cast<float>(unpack_low(byte)) * scale +
                 static_cast<float>(intermediate[input_base + column + 1]) *
                     static_cast<float>(unpack_high(byte)) * scale;
    }
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const c10::BFloat16 expert_value =
          static_cast<c10::BFloat16>(reduction[0]);
      const c10::BFloat16 weighted = static_cast<c10::BFloat16>(
          static_cast<float>(expert_value) *
          static_cast<float>(routing_weights[route]));
      route_output[route * kHiddenSize + output_feature] = weighted;
    }
    __syncthreads();
  }
}

// GEMM-like prefill path. A 16x16 block computes sixteen routes and sixteen
// output channels while sharing 32-wide activation and dequantized weight
// tiles. Routes are already grouped by expert, so every block has one expert.
__global__ void packed_q4_gate_up_tiled_kernel(
    const c10::BFloat16* __restrict__ hidden,
    const std::int64_t* __restrict__ route_order,
    const std::int64_t* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ gate_packed,
    const float* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_packed,
    const float* __restrict__ up_scales,
    c10::BFloat16* __restrict__ intermediate,
    std::int64_t group_size) {
  const int local_output = threadIdx.x;
  const int local_route = threadIdx.y;
  const int thread = local_route * kOutputTile + local_output;
  const std::int64_t expert = blockIdx.y;
  const std::int64_t route_base =
      expert_offsets[expert] + blockIdx.z * kRouteTile;
  if (route_base >= expert_offsets[expert + 1]) return;
  const std::int64_t route_rank =
      route_base + local_route;
  const std::int64_t route =
      route_rank < expert_offsets[expert + 1] ? route_order[route_rank] : -1;
  const std::int64_t output_feature =
      static_cast<std::int64_t>(blockIdx.x) * kOutputTile + local_output;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kIntermediateSize) * kHiddenSize;
  constexpr std::int64_t kPackedStride = kMatrixElements / 2;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  __shared__ float input_tile[kRouteTile][kReductionTile];
  __shared__ float gate_tile[kOutputTile][kReductionTile];
  __shared__ float up_tile[kOutputTile][kReductionTile];

  float gate_accumulator = 0.0f;
  float up_accumulator = 0.0f;
  for (int reduction_base = 0; reduction_base < kHiddenSize;
       reduction_base += kReductionTile) {
    for (int load = thread; load < kRouteTile * kReductionTile;
         load += kRouteTile * kOutputTile) {
      const int row = load / kReductionTile;
      const int column = load % kReductionTile;
      const std::int64_t input_rank =
          expert_offsets[expert] + blockIdx.z * kRouteTile + row;
      const std::int64_t input_route =
          input_rank < expert_offsets[expert + 1] ? route_order[input_rank] : -1;
      input_tile[row][column] =
          input_route >= 0
          ? static_cast<float>(
                hidden[(input_route / kTopK) * kHiddenSize + reduction_base +
                       column])
          : 0.0f;

      const std::int64_t weight_feature =
          static_cast<std::int64_t>(blockIdx.x) * kOutputTile + row;
      if (weight_feature < kIntermediateSize) {
        const std::int64_t weight_index =
            weight_feature * kHiddenSize + reduction_base + column;
        const std::uint8_t gate_byte =
            gate_packed[expert * kPackedStride + (weight_index >> 1)];
        const std::uint8_t up_byte =
            up_packed[expert * kPackedStride + (weight_index >> 1)];
        const int gate_quantized = (weight_index & 1) == 0
                                       ? unpack_low(gate_byte)
                                       : unpack_high(gate_byte);
        const int up_quantized = (weight_index & 1) == 0
                                     ? unpack_low(up_byte)
                                     : unpack_high(up_byte);
        gate_tile[row][column] =
            static_cast<float>(gate_quantized) *
            gate_scales[expert * scale_stride + weight_index / group_size];
        up_tile[row][column] =
            static_cast<float>(up_quantized) *
            up_scales[expert * scale_stride + weight_index / group_size];
      } else {
        gate_tile[row][column] = 0.0f;
        up_tile[row][column] = 0.0f;
      }
    }
    __syncthreads();
    if (route >= 0 && output_feature < kIntermediateSize) {
#pragma unroll
      for (int reduction = 0; reduction < kReductionTile; ++reduction) {
        const float input = input_tile[local_route][reduction];
        gate_accumulator += input * gate_tile[local_output][reduction];
        up_accumulator += input * up_tile[local_output][reduction];
      }
    }
    __syncthreads();
  }
  if (route >= 0 && output_feature < kIntermediateSize) {
    const c10::BFloat16 gate_bf16 =
        static_cast<c10::BFloat16>(gate_accumulator);
    const c10::BFloat16 up_bf16 =
        static_cast<c10::BFloat16>(up_accumulator);
    const float gate = static_cast<float>(gate_bf16);
    intermediate[route * kIntermediateSize + output_feature] =
        static_cast<c10::BFloat16>(
            gate / (1.0f + expf(-gate)) * static_cast<float>(up_bf16));
  }
}

__global__ void packed_q4_down_tiled_kernel(
    const c10::BFloat16* __restrict__ intermediate,
    const c10::BFloat16* __restrict__ routing_weights,
    const std::int64_t* __restrict__ route_order,
    const std::int64_t* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ down_packed,
    const float* __restrict__ down_scales,
    c10::BFloat16* __restrict__ route_output,
    std::int64_t group_size) {
  const int local_output = threadIdx.x;
  const int local_route = threadIdx.y;
  const int thread = local_route * kOutputTile + local_output;
  const std::int64_t expert = blockIdx.y;
  const std::int64_t route_base =
      expert_offsets[expert] + blockIdx.z * kRouteTile;
  if (route_base >= expert_offsets[expert + 1]) return;
  const std::int64_t route_rank =
      route_base + local_route;
  const std::int64_t route =
      route_rank < expert_offsets[expert + 1] ? route_order[route_rank] : -1;
  const std::int64_t output_feature =
      static_cast<std::int64_t>(blockIdx.x) * kOutputTile + local_output;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kHiddenSize) * kIntermediateSize;
  constexpr std::int64_t kPackedStride = kMatrixElements / 2;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  __shared__ float input_tile[kRouteTile][kReductionTile];
  __shared__ float weight_tile[kOutputTile][kReductionTile];

  float accumulator = 0.0f;
  for (int reduction_base = 0; reduction_base < kIntermediateSize;
       reduction_base += kReductionTile) {
    for (int load = thread; load < kRouteTile * kReductionTile;
         load += kRouteTile * kOutputTile) {
      const int row = load / kReductionTile;
      const int column = load % kReductionTile;
      const std::int64_t input_rank =
          expert_offsets[expert] + blockIdx.z * kRouteTile + row;
      const std::int64_t input_route =
          input_rank < expert_offsets[expert + 1] ? route_order[input_rank] : -1;
      input_tile[row][column] =
          input_route >= 0
          ? static_cast<float>(
                intermediate[input_route * kIntermediateSize + reduction_base +
                             column])
          : 0.0f;

      const std::int64_t weight_feature =
          static_cast<std::int64_t>(blockIdx.x) * kOutputTile + row;
      if (weight_feature < kHiddenSize) {
        const std::int64_t weight_index =
            weight_feature * kIntermediateSize + reduction_base + column;
        const std::uint8_t byte =
            down_packed[expert * kPackedStride + (weight_index >> 1)];
        const int quantized = (weight_index & 1) == 0 ? unpack_low(byte)
                                                       : unpack_high(byte);
        weight_tile[row][column] =
            static_cast<float>(quantized) *
            down_scales[expert * scale_stride + weight_index / group_size];
      } else {
        weight_tile[row][column] = 0.0f;
      }
    }
    __syncthreads();
    if (route >= 0 && output_feature < kHiddenSize) {
#pragma unroll
      for (int reduction = 0; reduction < kReductionTile; ++reduction) {
        accumulator += input_tile[local_route][reduction] *
                       weight_tile[local_output][reduction];
      }
    }
    __syncthreads();
  }
  if (route >= 0 && output_feature < kHiddenSize) {
    const c10::BFloat16 expert_value =
        static_cast<c10::BFloat16>(accumulator);
    route_output[route * kHiddenSize + output_feature] =
        static_cast<c10::BFloat16>(
            static_cast<float>(expert_value) *
            static_cast<float>(routing_weights[route]));
  }
}

// Vectorized prefill candidate. A 16x16 thread block computes a 16x64 output
// tile, so each thread accumulates four output channels while all four reuse
// the same activation tile. This cuts the number of expert/output blocks by 4.
__global__ void packed_q4_gate_up_vectorized_kernel(
    const c10::BFloat16* __restrict__ hidden,
    const std::int64_t* __restrict__ route_order,
    const std::int64_t* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ gate_packed,
    const float* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_packed,
    const float* __restrict__ up_scales,
    c10::BFloat16* __restrict__ intermediate,
    std::int64_t group_size) {
  const int local_output = threadIdx.x;
  const int local_route = threadIdx.y;
  const int thread = local_route * kOutputTile + local_output;
  __shared__ std::int64_t resolved_expert;
  __shared__ std::int64_t resolved_route_base;
  if (thread == 0) {
    resolve_compact_route_tile(
        expert_offsets, blockIdx.y, &resolved_expert, &resolved_route_base);
  }
  __syncthreads();
  if (resolved_expert < 0) return;
  const std::int64_t expert = resolved_expert;
  const std::int64_t route_base = resolved_route_base;
  const std::int64_t route_rank = route_base + local_route;
  const std::int64_t route =
      route_rank < expert_offsets[expert + 1] ? route_order[route_rank] : -1;
  const std::int64_t output_base =
      static_cast<std::int64_t>(blockIdx.x) * kVectorOutputTile;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kIntermediateSize) * kHiddenSize;
  constexpr std::int64_t kPackedStride = kMatrixElements / 2;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  __shared__ float input_tile[kRouteTile][kReductionTile];
  __shared__ float gate_tile[kVectorOutputTile][kReductionTile];
  __shared__ float up_tile[kVectorOutputTile][kReductionTile];

  float gate_accumulators[kOutputsPerThread] = {};
  float up_accumulators[kOutputsPerThread] = {};
  for (int reduction_base = 0; reduction_base < kHiddenSize;
       reduction_base += kReductionTile) {
    for (int load = thread; load < kRouteTile * kReductionTile;
         load += kRouteTile * kOutputTile) {
      const int row = load / kReductionTile;
      const int column = load % kReductionTile;
      const std::int64_t input_rank = route_base + row;
      const std::int64_t input_route =
          input_rank < expert_offsets[expert + 1] ? route_order[input_rank] : -1;
      input_tile[row][column] =
          input_route >= 0
              ? static_cast<float>(hidden[
                    (input_route / kTopK) * kHiddenSize + reduction_base + column])
              : 0.0f;
    }
    for (int load = thread; load < kVectorOutputTile * kReductionTile;
         load += kRouteTile * kOutputTile) {
      const int output = load / kReductionTile;
      const int column = load % kReductionTile;
      const std::int64_t output_feature = output_base + output;
      if (output_feature < kIntermediateSize) {
        const std::int64_t weight_index =
            output_feature * kHiddenSize + reduction_base + column;
        const std::uint8_t gate_byte =
            gate_packed[expert * kPackedStride + (weight_index >> 1)];
        const std::uint8_t up_byte =
            up_packed[expert * kPackedStride + (weight_index >> 1)];
        const int gate_quantized = (weight_index & 1) == 0
                                       ? unpack_low(gate_byte)
                                       : unpack_high(gate_byte);
        const int up_quantized = (weight_index & 1) == 0
                                     ? unpack_low(up_byte)
                                     : unpack_high(up_byte);
        gate_tile[output][column] =
            static_cast<float>(gate_quantized) *
            gate_scales[expert * scale_stride + weight_index / group_size];
        up_tile[output][column] =
            static_cast<float>(up_quantized) *
            up_scales[expert * scale_stride + weight_index / group_size];
      } else {
        gate_tile[output][column] = 0.0f;
        up_tile[output][column] = 0.0f;
      }
    }
    __syncthreads();
    if (route >= 0) {
#pragma unroll
      for (int output_slot = 0; output_slot < kOutputsPerThread; ++output_slot) {
        const int output = local_output + output_slot * kOutputTile;
#pragma unroll
        for (int reduction = 0; reduction < kReductionTile; ++reduction) {
          const float input = input_tile[local_route][reduction];
          gate_accumulators[output_slot] += input * gate_tile[output][reduction];
          up_accumulators[output_slot] += input * up_tile[output][reduction];
        }
      }
    }
    __syncthreads();
  }
  if (route >= 0) {
#pragma unroll
    for (int output_slot = 0; output_slot < kOutputsPerThread; ++output_slot) {
      const std::int64_t output_feature =
          output_base + local_output + output_slot * kOutputTile;
      if (output_feature < kIntermediateSize) {
        const c10::BFloat16 gate_bf16 =
            static_cast<c10::BFloat16>(gate_accumulators[output_slot]);
        const c10::BFloat16 up_bf16 =
            static_cast<c10::BFloat16>(up_accumulators[output_slot]);
        const float gate = static_cast<float>(gate_bf16);
        intermediate[route * kIntermediateSize + output_feature] =
            static_cast<c10::BFloat16>(
                gate / (1.0f + expf(-gate)) * static_cast<float>(up_bf16));
      }
    }
  }
}

__global__ void packed_q4_down_vectorized_kernel(
    const c10::BFloat16* __restrict__ intermediate,
    const c10::BFloat16* __restrict__ routing_weights,
    const std::int64_t* __restrict__ route_order,
    const std::int64_t* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ down_packed,
    const float* __restrict__ down_scales,
    c10::BFloat16* __restrict__ route_output,
    std::int64_t group_size) {
  const int local_output = threadIdx.x;
  const int local_route = threadIdx.y;
  const int thread = local_route * kOutputTile + local_output;
  __shared__ std::int64_t resolved_expert;
  __shared__ std::int64_t resolved_route_base;
  if (thread == 0) {
    resolve_compact_route_tile(
        expert_offsets, blockIdx.y, &resolved_expert, &resolved_route_base);
  }
  __syncthreads();
  if (resolved_expert < 0) return;
  const std::int64_t expert = resolved_expert;
  const std::int64_t route_base = resolved_route_base;
  const std::int64_t route_rank = route_base + local_route;
  const std::int64_t route =
      route_rank < expert_offsets[expert + 1] ? route_order[route_rank] : -1;
  const std::int64_t output_base =
      static_cast<std::int64_t>(blockIdx.x) * kVectorOutputTile;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kHiddenSize) * kIntermediateSize;
  constexpr std::int64_t kPackedStride = kMatrixElements / 2;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  __shared__ float input_tile[kRouteTile][kReductionTile];
  __shared__ float weight_tile[kVectorOutputTile][kReductionTile];

  float accumulators[kOutputsPerThread] = {};
  for (int reduction_base = 0; reduction_base < kIntermediateSize;
       reduction_base += kReductionTile) {
    for (int load = thread; load < kRouteTile * kReductionTile;
         load += kRouteTile * kOutputTile) {
      const int row = load / kReductionTile;
      const int column = load % kReductionTile;
      const std::int64_t input_rank = route_base + row;
      const std::int64_t input_route =
          input_rank < expert_offsets[expert + 1] ? route_order[input_rank] : -1;
      input_tile[row][column] =
          input_route >= 0
              ? static_cast<float>(intermediate[
                    input_route * kIntermediateSize + reduction_base + column])
              : 0.0f;
    }
    for (int load = thread; load < kVectorOutputTile * kReductionTile;
         load += kRouteTile * kOutputTile) {
      const int output = load / kReductionTile;
      const int column = load % kReductionTile;
      const std::int64_t output_feature = output_base + output;
      if (output_feature < kHiddenSize) {
        const std::int64_t weight_index =
            output_feature * kIntermediateSize + reduction_base + column;
        const std::uint8_t byte =
            down_packed[expert * kPackedStride + (weight_index >> 1)];
        const int quantized = (weight_index & 1) == 0 ? unpack_low(byte)
                                                       : unpack_high(byte);
        weight_tile[output][column] =
            static_cast<float>(quantized) *
            down_scales[expert * scale_stride + weight_index / group_size];
      } else {
        weight_tile[output][column] = 0.0f;
      }
    }
    __syncthreads();
    if (route >= 0) {
#pragma unroll
      for (int output_slot = 0; output_slot < kOutputsPerThread; ++output_slot) {
        const int output = local_output + output_slot * kOutputTile;
#pragma unroll
        for (int reduction = 0; reduction < kReductionTile; ++reduction) {
          accumulators[output_slot] +=
              input_tile[local_route][reduction] * weight_tile[output][reduction];
        }
      }
    }
    __syncthreads();
  }
  if (route >= 0) {
#pragma unroll
    for (int output_slot = 0; output_slot < kOutputsPerThread; ++output_slot) {
      const std::int64_t output_feature =
          output_base + local_output + output_slot * kOutputTile;
      if (output_feature < kHiddenSize) {
        const c10::BFloat16 expert_value =
            static_cast<c10::BFloat16>(accumulators[output_slot]);
        route_output[route * kHiddenSize + output_feature] =
            static_cast<c10::BFloat16>(
                static_cast<float>(expert_value) *
                static_cast<float>(routing_weights[route]));
      }
    }
  }
}

// Direct symmetric Q8 kernels share the route schedule with packed Q4 but
// consume one signed byte per weight.  Q8 bytes remain compressed throughout
// execution; no BF16/F32 expert matrix is materialized or cached.
__global__ void packed_q8_gate_up_swiglu_kernel(
    const c10::BFloat16* __restrict__ hidden,
    const std::int64_t* __restrict__ expert_indices,
    const std::int8_t* __restrict__ gate_quantized,
    const float* __restrict__ gate_scales,
    const std::int8_t* __restrict__ up_quantized,
    const float* __restrict__ up_scales,
    c10::BFloat16* __restrict__ intermediate,
    std::int64_t rows,
    std::int64_t group_size) {
  const std::int64_t feature = blockIdx.x;
  const std::int64_t route = blockIdx.y;
  const std::int64_t route_count = rows * kTopK;
  if (route >= route_count || feature >= kIntermediateSize) return;

  const std::int64_t token = route / kTopK;
  const std::int64_t expert = expert_indices[route];
  if (expert < 0 || expert >= kExpertCount) return;
  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kIntermediateSize) * kHiddenSize;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  const std::int64_t weight_row = feature * kHiddenSize;
  const std::int64_t data_base = expert * kMatrixElements + weight_row;
  const std::int64_t scale_base =
      expert * scale_stride + weight_row / group_size;
  const std::int64_t input_base = token * kHiddenSize;

  float gate_partial = 0.0f;
  float up_partial = 0.0f;
  for (std::int64_t column = threadIdx.x; column < kHiddenSize;
       column += blockDim.x) {
    const float input = static_cast<float>(hidden[input_base + column]);
    const std::int64_t scale_index = scale_base + column / group_size;
    gate_partial += input * static_cast<float>(gate_quantized[data_base + column]) *
                    gate_scales[scale_index];
    up_partial += input * static_cast<float>(up_quantized[data_base + column]) *
                  up_scales[scale_index];
  }
  __shared__ float gate_reduction[kThreads];
  __shared__ float up_reduction[kThreads];
  gate_reduction[threadIdx.x] = gate_partial;
  up_reduction[threadIdx.x] = up_partial;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      gate_reduction[threadIdx.x] += gate_reduction[threadIdx.x + stride];
      up_reduction[threadIdx.x] += up_reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const c10::BFloat16 gate_bf16 =
        static_cast<c10::BFloat16>(gate_reduction[0]);
    const c10::BFloat16 up_bf16 =
        static_cast<c10::BFloat16>(up_reduction[0]);
    const float gate = static_cast<float>(gate_bf16);
    intermediate[route * kIntermediateSize + feature] =
        static_cast<c10::BFloat16>(
            gate / (1.0f + expf(-gate)) * static_cast<float>(up_bf16));
  }
}

__global__ void packed_q8_down_route_kernel(
    const c10::BFloat16* __restrict__ intermediate,
    const std::int64_t* __restrict__ expert_indices,
    const c10::BFloat16* __restrict__ routing_weights,
    const std::int8_t* __restrict__ down_quantized,
    const float* __restrict__ down_scales,
    c10::BFloat16* __restrict__ output,
    std::int64_t rows,
    std::int64_t group_size) {
  const std::int64_t output_feature = blockIdx.x;
  const std::int64_t token = blockIdx.y;
  if (token >= rows || output_feature >= kHiddenSize) return;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kHiddenSize) * kIntermediateSize;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  const std::int64_t weight_row = output_feature * kIntermediateSize;
  float partial = 0.0f;
  for (int slot = 0; slot < kTopK; ++slot) {
    const std::int64_t route = token * kTopK + slot;
    const std::int64_t expert = expert_indices[route];
    if (expert < 0 || expert >= kExpertCount) continue;
    const std::int64_t data_base = expert * kMatrixElements + weight_row;
    const std::int64_t scale_base =
        expert * scale_stride + weight_row / group_size;
    const std::int64_t input_base = route * kIntermediateSize;
    float expert_partial = 0.0f;
    for (std::int64_t column = threadIdx.x; column < kIntermediateSize;
         column += blockDim.x) {
      expert_partial +=
          static_cast<float>(intermediate[input_base + column]) *
          static_cast<float>(down_quantized[data_base + column]) *
          down_scales[scale_base + column / group_size];
    }
    partial += static_cast<float>(routing_weights[route]) * expert_partial;
  }
  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[token * kHiddenSize + output_feature] =
        static_cast<c10::BFloat16>(reduction[0]);
  }
}

__global__ void packed_q8_gate_up_vectorized_kernel(
    const c10::BFloat16* __restrict__ hidden,
    const std::int64_t* __restrict__ route_order,
    const std::int64_t* __restrict__ expert_offsets,
    const std::int8_t* __restrict__ gate_quantized,
    const float* __restrict__ gate_scales,
    const std::int8_t* __restrict__ up_quantized,
    const float* __restrict__ up_scales,
    c10::BFloat16* __restrict__ intermediate,
    std::int64_t group_size) {
  const int local_output = threadIdx.x;
  const int local_route = threadIdx.y;
  const int thread = local_route * kOutputTile + local_output;
  __shared__ std::int64_t resolved_expert;
  __shared__ std::int64_t resolved_route_base;
  if (thread == 0) {
    resolve_compact_route_tile(
        expert_offsets, blockIdx.y, &resolved_expert, &resolved_route_base);
  }
  __syncthreads();
  if (resolved_expert < 0) return;
  const std::int64_t expert = resolved_expert;
  const std::int64_t route_base = resolved_route_base;
  const std::int64_t route_rank = route_base + local_route;
  const std::int64_t route =
      route_rank < expert_offsets[expert + 1] ? route_order[route_rank] : -1;
  const std::int64_t output_base =
      static_cast<std::int64_t>(blockIdx.x) * kVectorOutputTile;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kIntermediateSize) * kHiddenSize;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  __shared__ float input_tile[kRouteTile][kReductionTile];
  __shared__ float gate_tile[kVectorOutputTile][kReductionTile];
  __shared__ float up_tile[kVectorOutputTile][kReductionTile];
  float gate_accumulators[kOutputsPerThread] = {};
  float up_accumulators[kOutputsPerThread] = {};

  for (int reduction_base = 0; reduction_base < kHiddenSize;
       reduction_base += kReductionTile) {
    for (int load = thread; load < kRouteTile * kReductionTile;
         load += kRouteTile * kOutputTile) {
      const int row = load / kReductionTile;
      const int column = load % kReductionTile;
      const std::int64_t input_rank = route_base + row;
      const std::int64_t input_route =
          input_rank < expert_offsets[expert + 1] ? route_order[input_rank] : -1;
      input_tile[row][column] =
          input_route >= 0
              ? static_cast<float>(hidden[
                    (input_route / kTopK) * kHiddenSize + reduction_base + column])
              : 0.0f;
    }
    for (int load = thread; load < kVectorOutputTile * kReductionTile;
         load += kRouteTile * kOutputTile) {
      const int output = load / kReductionTile;
      const int column = load % kReductionTile;
      const std::int64_t output_feature = output_base + output;
      if (output_feature < kIntermediateSize) {
        const std::int64_t weight_index =
            output_feature * kHiddenSize + reduction_base + column;
        const std::int64_t data_index = expert * kMatrixElements + weight_index;
        const std::int64_t scale_index =
            expert * scale_stride + weight_index / group_size;
        gate_tile[output][column] =
            static_cast<float>(gate_quantized[data_index]) *
            gate_scales[scale_index];
        up_tile[output][column] =
            static_cast<float>(up_quantized[data_index]) * up_scales[scale_index];
      } else {
        gate_tile[output][column] = 0.0f;
        up_tile[output][column] = 0.0f;
      }
    }
    __syncthreads();
    if (route >= 0) {
#pragma unroll
      for (int output_slot = 0; output_slot < kOutputsPerThread; ++output_slot) {
        const int output = local_output + output_slot * kOutputTile;
#pragma unroll
        for (int reduction = 0; reduction < kReductionTile; ++reduction) {
          const float input = input_tile[local_route][reduction];
          gate_accumulators[output_slot] += input * gate_tile[output][reduction];
          up_accumulators[output_slot] += input * up_tile[output][reduction];
        }
      }
    }
    __syncthreads();
  }
  if (route >= 0) {
#pragma unroll
    for (int output_slot = 0; output_slot < kOutputsPerThread; ++output_slot) {
      const std::int64_t output_feature =
          output_base + local_output + output_slot * kOutputTile;
      if (output_feature < kIntermediateSize) {
        const c10::BFloat16 gate_bf16 =
            static_cast<c10::BFloat16>(gate_accumulators[output_slot]);
        const c10::BFloat16 up_bf16 =
            static_cast<c10::BFloat16>(up_accumulators[output_slot]);
        const float gate = static_cast<float>(gate_bf16);
        intermediate[route * kIntermediateSize + output_feature] =
            static_cast<c10::BFloat16>(
                gate / (1.0f + expf(-gate)) * static_cast<float>(up_bf16));
      }
    }
  }
}

__global__ void packed_q8_down_vectorized_kernel(
    const c10::BFloat16* __restrict__ intermediate,
    const c10::BFloat16* __restrict__ routing_weights,
    const std::int64_t* __restrict__ route_order,
    const std::int64_t* __restrict__ expert_offsets,
    const std::int8_t* __restrict__ down_quantized,
    const float* __restrict__ down_scales,
    c10::BFloat16* __restrict__ route_output,
    std::int64_t group_size) {
  const int local_output = threadIdx.x;
  const int local_route = threadIdx.y;
  const int thread = local_route * kOutputTile + local_output;
  __shared__ std::int64_t resolved_expert;
  __shared__ std::int64_t resolved_route_base;
  if (thread == 0) {
    resolve_compact_route_tile(
        expert_offsets, blockIdx.y, &resolved_expert, &resolved_route_base);
  }
  __syncthreads();
  if (resolved_expert < 0) return;
  const std::int64_t expert = resolved_expert;
  const std::int64_t route_base = resolved_route_base;
  const std::int64_t route_rank = route_base + local_route;
  const std::int64_t route =
      route_rank < expert_offsets[expert + 1] ? route_order[route_rank] : -1;
  const std::int64_t output_base =
      static_cast<std::int64_t>(blockIdx.x) * kVectorOutputTile;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kHiddenSize) * kIntermediateSize;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  __shared__ float input_tile[kRouteTile][kReductionTile];
  __shared__ float weight_tile[kVectorOutputTile][kReductionTile];
  float accumulators[kOutputsPerThread] = {};

  for (int reduction_base = 0; reduction_base < kIntermediateSize;
       reduction_base += kReductionTile) {
    for (int load = thread; load < kRouteTile * kReductionTile;
         load += kRouteTile * kOutputTile) {
      const int row = load / kReductionTile;
      const int column = load % kReductionTile;
      const std::int64_t input_rank = route_base + row;
      const std::int64_t input_route =
          input_rank < expert_offsets[expert + 1] ? route_order[input_rank] : -1;
      input_tile[row][column] =
          input_route >= 0
              ? static_cast<float>(intermediate[
                    input_route * kIntermediateSize + reduction_base + column])
              : 0.0f;
    }
    for (int load = thread; load < kVectorOutputTile * kReductionTile;
         load += kRouteTile * kOutputTile) {
      const int output = load / kReductionTile;
      const int column = load % kReductionTile;
      const std::int64_t output_feature = output_base + output;
      if (output_feature < kHiddenSize) {
        const std::int64_t weight_index =
            output_feature * kIntermediateSize + reduction_base + column;
        const std::int64_t data_index = expert * kMatrixElements + weight_index;
        const std::int64_t scale_index =
            expert * scale_stride + weight_index / group_size;
        weight_tile[output][column] =
            static_cast<float>(down_quantized[data_index]) *
            down_scales[scale_index];
      } else {
        weight_tile[output][column] = 0.0f;
      }
    }
    __syncthreads();
    if (route >= 0) {
#pragma unroll
      for (int output_slot = 0; output_slot < kOutputsPerThread; ++output_slot) {
        const int output = local_output + output_slot * kOutputTile;
#pragma unroll
        for (int reduction = 0; reduction < kReductionTile; ++reduction) {
          accumulators[output_slot] +=
              input_tile[local_route][reduction] * weight_tile[output][reduction];
        }
      }
    }
    __syncthreads();
  }
  if (route >= 0) {
#pragma unroll
    for (int output_slot = 0; output_slot < kOutputsPerThread; ++output_slot) {
      const std::int64_t output_feature =
          output_base + local_output + output_slot * kOutputTile;
      if (output_feature < kHiddenSize) {
        const c10::BFloat16 expert_value =
            static_cast<c10::BFloat16>(accumulators[output_slot]);
        route_output[route * kHiddenSize + output_feature] =
            static_cast<c10::BFloat16>(
                static_cast<float>(expert_value) *
                static_cast<float>(routing_weights[route]));
      }
    }
  }
}

#if defined(__HIP_PLATFORM_AMD__) && defined(UMA_QMOE_USE_WMMA)

// gfx11 wave32 path. Each wave computes a 16-route x 16-output tile with
// native BF16 WMMA. Canonical Q4 weights are decoded directly into the current
// K=16 shared-memory fragment; no full dequantized weight tensor is materialized.
__global__ __launch_bounds__(32) void packed_q4_gate_up_wmma_kernel(
    const c10::BFloat16* __restrict__ hidden,
    const std::int64_t* __restrict__ route_order,
    const std::int64_t* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ gate_packed,
    const float* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_packed,
    const float* __restrict__ up_scales,
    c10::BFloat16* __restrict__ intermediate,
    std::int64_t group_size) {
  using BFloat16 = rocwmma::bfloat16_t;
  using FragmentA = rocwmma::fragment<
      rocwmma::matrix_a,
      16,
      16,
      16,
      BFloat16,
      rocwmma::row_major>;
  using FragmentB = rocwmma::fragment<
      rocwmma::matrix_b,
      16,
      16,
      16,
      BFloat16,
      rocwmma::row_major>;
  using FragmentAccumulator =
      rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

  const int lane = threadIdx.x;
  const std::int64_t expert = blockIdx.y;
  const std::int64_t route_base =
      expert_offsets[expert] + blockIdx.z * kRouteTile;
  if (route_base >= expert_offsets[expert + 1]) return;
  const std::int64_t output_base =
      static_cast<std::int64_t>(blockIdx.x) * kOutputTile;
  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kIntermediateSize) * kHiddenSize;
  constexpr std::int64_t kPackedStride = kMatrixElements / 2;
  const std::int64_t scale_stride = kMatrixElements / group_size;

  __shared__ BFloat16 input_tile[kRouteTile * 16];
  __shared__ BFloat16 gate_tile[16 * kOutputTile];
  __shared__ BFloat16 up_tile[16 * kOutputTile];
  __shared__ float gate_output[kRouteTile * kOutputTile];
  __shared__ float up_output[kRouteTile * kOutputTile];

  FragmentAccumulator gate_accumulator;
  FragmentAccumulator up_accumulator;
  rocwmma::fill_fragment(gate_accumulator, 0.0f);
  rocwmma::fill_fragment(up_accumulator, 0.0f);

  for (int reduction_base = 0; reduction_base < kHiddenSize;
       reduction_base += 16) {
    for (int index = lane; index < kRouteTile * 16; index += 32) {
      const int route_row = index / 16;
      const int reduction = index % 16;
      const std::int64_t route_rank = route_base + route_row;
      const std::int64_t route =
          route_rank < expert_offsets[expert + 1] ? route_order[route_rank] : -1;
      input_tile[index] = route >= 0
                              ? BFloat16(static_cast<float>(hidden[
                                    (route / kTopK) * kHiddenSize +
                                    reduction_base + reduction]))
                              : BFloat16(0.0f);

      const int weight_k = index / kOutputTile;
      const int output_column = index % kOutputTile;
      const std::int64_t output_feature = output_base + output_column;
      if (output_feature < kIntermediateSize) {
        const std::int64_t weight_index =
            output_feature * kHiddenSize + reduction_base + weight_k;
        const std::uint8_t gate_byte =
            gate_packed[expert * kPackedStride + (weight_index >> 1)];
        const std::uint8_t up_byte =
            up_packed[expert * kPackedStride + (weight_index >> 1)];
        const int gate_quantized = (weight_index & 1) == 0
                                       ? unpack_low(gate_byte)
                                       : unpack_high(gate_byte);
        const int up_quantized = (weight_index & 1) == 0
                                     ? unpack_low(up_byte)
                                     : unpack_high(up_byte);
        gate_tile[index] = BFloat16(
            static_cast<float>(gate_quantized) *
            gate_scales[expert * scale_stride + weight_index / group_size]);
        up_tile[index] = BFloat16(
            static_cast<float>(up_quantized) *
            up_scales[expert * scale_stride + weight_index / group_size]);
      } else {
        gate_tile[index] = BFloat16(0.0f);
        up_tile[index] = BFloat16(0.0f);
      }
    }
    __syncthreads();
    FragmentA input_fragment;
    FragmentB gate_fragment;
    FragmentB up_fragment;
    rocwmma::load_matrix_sync(input_fragment, input_tile, 16);
    rocwmma::load_matrix_sync(gate_fragment, gate_tile, kOutputTile);
    rocwmma::load_matrix_sync(up_fragment, up_tile, kOutputTile);
    rocwmma::mma_sync(
        gate_accumulator, input_fragment, gate_fragment, gate_accumulator);
    rocwmma::mma_sync(
        up_accumulator, input_fragment, up_fragment, up_accumulator);
    __syncthreads();
  }

  rocwmma::store_matrix_sync(
      gate_output,
      gate_accumulator,
      kOutputTile,
      rocwmma::mem_row_major);
  rocwmma::store_matrix_sync(
      up_output, up_accumulator, kOutputTile, rocwmma::mem_row_major);
  __syncthreads();
  for (int index = lane; index < kRouteTile * kOutputTile; index += 32) {
    const int route_row = index / kOutputTile;
    const int output_column = index % kOutputTile;
    const std::int64_t route_rank = route_base + route_row;
    const std::int64_t route =
        route_rank < expert_offsets[expert + 1] ? route_order[route_rank] : -1;
    const std::int64_t output_feature = output_base + output_column;
    if (route >= 0 && output_feature < kIntermediateSize) {
      const c10::BFloat16 gate_bf16 =
          static_cast<c10::BFloat16>(gate_output[index]);
      const c10::BFloat16 up_bf16 =
          static_cast<c10::BFloat16>(up_output[index]);
      const float gate = static_cast<float>(gate_bf16);
      intermediate[route * kIntermediateSize + output_feature] =
          static_cast<c10::BFloat16>(
              gate / (1.0f + expf(-gate)) * static_cast<float>(up_bf16));
    }
  }
}

__global__ __launch_bounds__(32) void packed_q4_down_wmma_kernel(
    const c10::BFloat16* __restrict__ intermediate,
    const c10::BFloat16* __restrict__ routing_weights,
    const std::int64_t* __restrict__ route_order,
    const std::int64_t* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ down_packed,
    const float* __restrict__ down_scales,
    c10::BFloat16* __restrict__ route_output,
    std::int64_t group_size) {
  using BFloat16 = rocwmma::bfloat16_t;
  using FragmentA = rocwmma::fragment<
      rocwmma::matrix_a,
      16,
      16,
      16,
      BFloat16,
      rocwmma::row_major>;
  using FragmentB = rocwmma::fragment<
      rocwmma::matrix_b,
      16,
      16,
      16,
      BFloat16,
      rocwmma::row_major>;
  using FragmentAccumulator =
      rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

  const int lane = threadIdx.x;
  const std::int64_t expert = blockIdx.y;
  const std::int64_t route_base =
      expert_offsets[expert] + blockIdx.z * kRouteTile;
  if (route_base >= expert_offsets[expert + 1]) return;
  const std::int64_t output_base =
      static_cast<std::int64_t>(blockIdx.x) * kOutputTile;
  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kHiddenSize) * kIntermediateSize;
  constexpr std::int64_t kPackedStride = kMatrixElements / 2;
  const std::int64_t scale_stride = kMatrixElements / group_size;

  __shared__ BFloat16 input_tile[kRouteTile * 16];
  __shared__ BFloat16 weight_tile[16 * kOutputTile];
  __shared__ float output_tile[kRouteTile * kOutputTile];

  FragmentAccumulator accumulator;
  rocwmma::fill_fragment(accumulator, 0.0f);
  for (int reduction_base = 0; reduction_base < kIntermediateSize;
       reduction_base += 16) {
    for (int index = lane; index < kRouteTile * 16; index += 32) {
      const int route_row = index / 16;
      const int reduction = index % 16;
      const std::int64_t route_rank = route_base + route_row;
      const std::int64_t route =
          route_rank < expert_offsets[expert + 1] ? route_order[route_rank] : -1;
      input_tile[index] = route >= 0
                              ? BFloat16(static_cast<float>(intermediate[
                                    route * kIntermediateSize + reduction_base +
                                    reduction]))
                              : BFloat16(0.0f);

      const int weight_k = index / kOutputTile;
      const int output_column = index % kOutputTile;
      const std::int64_t output_feature = output_base + output_column;
      if (output_feature < kHiddenSize) {
        const std::int64_t weight_index =
            output_feature * kIntermediateSize + reduction_base + weight_k;
        const std::uint8_t byte =
            down_packed[expert * kPackedStride + (weight_index >> 1)];
        const int quantized = (weight_index & 1) == 0 ? unpack_low(byte)
                                                       : unpack_high(byte);
        weight_tile[index] = BFloat16(
            static_cast<float>(quantized) *
            down_scales[expert * scale_stride + weight_index / group_size]);
      } else {
        weight_tile[index] = BFloat16(0.0f);
      }
    }
    __syncthreads();
    FragmentA input_fragment;
    FragmentB weight_fragment;
    rocwmma::load_matrix_sync(input_fragment, input_tile, 16);
    rocwmma::load_matrix_sync(weight_fragment, weight_tile, kOutputTile);
    rocwmma::mma_sync(
        accumulator, input_fragment, weight_fragment, accumulator);
    __syncthreads();
  }

  rocwmma::store_matrix_sync(
      output_tile, accumulator, kOutputTile, rocwmma::mem_row_major);
  __syncthreads();
  for (int index = lane; index < kRouteTile * kOutputTile; index += 32) {
    const int route_row = index / kOutputTile;
    const int output_column = index % kOutputTile;
    const std::int64_t route_rank = route_base + route_row;
    const std::int64_t route =
        route_rank < expert_offsets[expert + 1] ? route_order[route_rank] : -1;
    const std::int64_t output_feature = output_base + output_column;
    if (route >= 0 && output_feature < kHiddenSize) {
      const c10::BFloat16 expert_value =
          static_cast<c10::BFloat16>(output_tile[index]);
      route_output[route * kHiddenSize + output_feature] =
          static_cast<c10::BFloat16>(
              static_cast<float>(expert_value) *
              static_cast<float>(routing_weights[route]));
    }
  }
}

#endif

__global__ void reduce_topk_route_output_kernel(
    const c10::BFloat16* __restrict__ route_output,
    c10::BFloat16* __restrict__ output,
    std::int64_t rows) {
  const std::int64_t token = blockIdx.y;
  const std::int64_t output_feature =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= rows || output_feature >= kHiddenSize) return;
  float value = 0.0f;
  for (int slot = 0; slot < kTopK; ++slot) {
    const std::int64_t route = token * kTopK + slot;
    value += static_cast<float>(
        route_output[route * kHiddenSize + output_feature]);
  }
  output[token * kHiddenSize + output_feature] =
      static_cast<c10::BFloat16>(value);
}

// One block computes one final hidden element and consumes all Top-8 routes,
// eliminating Python expert grouping, index_add, and atomics.
__global__ void packed_q4_down_route_kernel(
    const c10::BFloat16* __restrict__ intermediate,
    const std::int64_t* __restrict__ expert_indices,
    const c10::BFloat16* __restrict__ routing_weights,
    const std::uint8_t* __restrict__ down_packed,
    const float* __restrict__ down_scales,
    c10::BFloat16* __restrict__ output,
    std::int64_t rows,
    std::int64_t group_size) {
  const std::int64_t output_feature = blockIdx.x;
  const std::int64_t token = blockIdx.y;
  if (token >= rows || output_feature >= kHiddenSize) return;

  constexpr std::int64_t kMatrixElements =
      static_cast<std::int64_t>(kHiddenSize) * kIntermediateSize;
  constexpr std::int64_t kPackedStride = kMatrixElements / 2;
  const std::int64_t scale_stride = kMatrixElements / group_size;
  const std::int64_t weight_row = output_feature * kIntermediateSize;
  float partial = 0.0f;

  for (int slot = 0; slot < kTopK; ++slot) {
    const std::int64_t route = token * kTopK + slot;
    const std::int64_t expert = expert_indices[route];
    if (expert < 0 || expert >= kExpertCount) continue;
    const float route_weight = static_cast<float>(routing_weights[route]);
    const std::int64_t packed_base = expert * kPackedStride + weight_row / 2;
    const std::int64_t scale_base =
        expert * scale_stride + weight_row / group_size;
    const std::int64_t input_base = route * kIntermediateSize;
    float expert_partial = 0.0f;
    for (std::int64_t byte_column = threadIdx.x;
         byte_column < kIntermediateSize / 2; byte_column += blockDim.x) {
      const std::int64_t column = byte_column * 2;
      const std::uint8_t byte = down_packed[packed_base + byte_column];
      const float scale = down_scales[scale_base + column / group_size];
      expert_partial +=
          static_cast<float>(intermediate[input_base + column]) *
              static_cast<float>(unpack_low(byte)) * scale +
          static_cast<float>(intermediate[input_base + column + 1]) *
              static_cast<float>(unpack_high(byte)) * scale;
    }
    partial += route_weight * expert_partial;
  }

  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[token * kHiddenSize + output_feature] =
        static_cast<c10::BFloat16>(reduction[0]);
  }
}

}  // namespace

torch::Tensor uma_qmoe_q4_linear_cuda(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    std::int64_t output_features,
    std::int64_t input_features,
    std::int64_t group_size) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty(
      {input.size(0), output_features}, input.options().dtype(torch::kBFloat16));
  const dim3 grid(
      static_cast<unsigned int>(output_features),
      static_cast<unsigned int>(input.size(0)),
      1);
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  packed_q4_bf16_gemv_kernel<<<grid, kThreads, 0, stream.stream()>>>(
      input.data_ptr<c10::BFloat16>(),
      packed.data_ptr<std::uint8_t>(),
      scales.data_ptr<float>(),
      output.data_ptr<c10::BFloat16>(),
      input.size(0),
      output_features,
      input_features,
      group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor uma_qmoe_q4_moe_forward_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& expert_indices,
    const torch::Tensor& routing_weights,
    const torch::Tensor& gate_packed,
    const torch::Tensor& gate_scales,
    const torch::Tensor& up_packed,
    const torch::Tensor& up_scales,
    const torch::Tensor& down_packed,
    const torch::Tensor& down_scales,
    std::int64_t group_size) {
  const c10::cuda::CUDAGuard device_guard(hidden.device());
  const auto stream = at::cuda::getCurrentCUDAStream(hidden.get_device());
  const std::int64_t rows = hidden.size(0);
  auto intermediate = torch::empty(
      {rows * kTopK, kIntermediateSize},
      hidden.options().dtype(torch::kBFloat16));
  auto output =
      torch::empty({rows, kHiddenSize}, hidden.options().dtype(torch::kBFloat16));

  const dim3 gate_up_grid(
      kIntermediateSize, static_cast<unsigned int>(rows * kTopK), 1);
  packed_q4_gate_up_swiglu_kernel<<<gate_up_grid, kThreads, 0, stream.stream()>>>(
      hidden.data_ptr<c10::BFloat16>(),
      expert_indices.data_ptr<std::int64_t>(),
      nullptr,
      gate_packed.data_ptr<std::uint8_t>(),
      gate_scales.data_ptr<float>(),
      up_packed.data_ptr<std::uint8_t>(),
      up_scales.data_ptr<float>(),
      intermediate.data_ptr<c10::BFloat16>(),
      rows,
      group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 down_grid(kHiddenSize, static_cast<unsigned int>(rows), 1);
  packed_q4_down_route_kernel<<<down_grid, kThreads, 0, stream.stream()>>>(
      intermediate.data_ptr<c10::BFloat16>(),
      expert_indices.data_ptr<std::int64_t>(),
      routing_weights.data_ptr<c10::BFloat16>(),
      down_packed.data_ptr<std::uint8_t>(),
      down_scales.data_ptr<float>(),
      output.data_ptr<c10::BFloat16>(),
      rows,
      group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor uma_qmoe_q4_moe_prefill_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& expert_indices,
    const torch::Tensor& routing_weights,
    const torch::Tensor& route_order,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& gate_packed,
    const torch::Tensor& gate_scales,
    const torch::Tensor& up_packed,
    const torch::Tensor& up_scales,
    const torch::Tensor& down_packed,
    const torch::Tensor& down_scales,
    std::int64_t group_size) {
  const c10::cuda::CUDAGuard device_guard(hidden.device());
  const auto stream = at::cuda::getCurrentCUDAStream(hidden.get_device());
  const std::int64_t rows = hidden.size(0);
  const std::int64_t route_count = rows * kTopK;
  auto intermediate = torch::empty(
      {route_count, kIntermediateSize},
      hidden.options().dtype(torch::kBFloat16));
  auto route_output = torch::empty(
      {route_count, kHiddenSize}, hidden.options().dtype(torch::kBFloat16));
  auto output =
      torch::empty({rows, kHiddenSize}, hidden.options().dtype(torch::kBFloat16));

  const unsigned int route_tiles =
      static_cast<unsigned int>((rows + kRouteTile - 1) / kRouteTile);
  const unsigned int compact_route_tiles = static_cast<unsigned int>(
      (route_count + kRouteTile - 1) / kRouteTile + kExpertCount - 1);
#if defined(__HIP_PLATFORM_AMD__) && defined(UMA_QMOE_USE_WMMA)
  const dim3 gate_up_grid(
      (kIntermediateSize + kOutputTile - 1) / kOutputTile,
      kExpertCount,
      route_tiles);
  packed_q4_gate_up_wmma_kernel<<<
      gate_up_grid, 32, 0, stream.stream()>>>(
      hidden.data_ptr<c10::BFloat16>(),
      route_order.data_ptr<std::int64_t>(),
      expert_offsets.data_ptr<std::int64_t>(),
      gate_packed.data_ptr<std::uint8_t>(),
      gate_scales.data_ptr<float>(),
      up_packed.data_ptr<std::uint8_t>(),
      up_scales.data_ptr<float>(),
      intermediate.data_ptr<c10::BFloat16>(),
      group_size);
#else
  const dim3 tiled_block(kOutputTile, kRouteTile, 1);
  const dim3 gate_up_grid(
      (kIntermediateSize + kVectorOutputTile - 1) / kVectorOutputTile,
      compact_route_tiles,
      1);
  packed_q4_gate_up_vectorized_kernel<<<
      gate_up_grid, tiled_block, 0, stream.stream()>>>(
      hidden.data_ptr<c10::BFloat16>(),
      route_order.data_ptr<std::int64_t>(),
      expert_offsets.data_ptr<std::int64_t>(),
      gate_packed.data_ptr<std::uint8_t>(),
      gate_scales.data_ptr<float>(),
      up_packed.data_ptr<std::uint8_t>(),
      up_scales.data_ptr<float>(),
      intermediate.data_ptr<c10::BFloat16>(),
      group_size);
#endif
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#if defined(__HIP_PLATFORM_AMD__) && defined(UMA_QMOE_USE_WMMA)
  const dim3 down_grid(
      (kHiddenSize + kOutputTile - 1) / kOutputTile,
      kExpertCount,
      route_tiles);
  packed_q4_down_wmma_kernel<<<down_grid, 32, 0, stream.stream()>>>(
      intermediate.data_ptr<c10::BFloat16>(),
      routing_weights.data_ptr<c10::BFloat16>(),
      route_order.data_ptr<std::int64_t>(),
      expert_offsets.data_ptr<std::int64_t>(),
      down_packed.data_ptr<std::uint8_t>(),
      down_scales.data_ptr<float>(),
      route_output.data_ptr<c10::BFloat16>(),
      group_size);
#else
  const dim3 down_grid(
      (kHiddenSize + kVectorOutputTile - 1) / kVectorOutputTile,
      compact_route_tiles,
      1);
  packed_q4_down_vectorized_kernel<<<down_grid, tiled_block, 0, stream.stream()>>>(
      intermediate.data_ptr<c10::BFloat16>(),
      routing_weights.data_ptr<c10::BFloat16>(),
      route_order.data_ptr<std::int64_t>(),
      expert_offsets.data_ptr<std::int64_t>(),
      down_packed.data_ptr<std::uint8_t>(),
      down_scales.data_ptr<float>(),
      route_output.data_ptr<c10::BFloat16>(),
      group_size);
#endif
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 reduce_grid(
      (kHiddenSize + kThreads - 1) / kThreads,
      static_cast<unsigned int>(rows),
      1);
  reduce_topk_route_output_kernel<<<reduce_grid, kThreads, 0, stream.stream()>>>(
      route_output.data_ptr<c10::BFloat16>(),
      output.data_ptr<c10::BFloat16>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor uma_qmoe_q8_moe_forward_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& expert_indices,
    const torch::Tensor& routing_weights,
    const torch::Tensor& gate_quantized,
    const torch::Tensor& gate_scales,
    const torch::Tensor& up_quantized,
    const torch::Tensor& up_scales,
    const torch::Tensor& down_quantized,
    const torch::Tensor& down_scales,
    std::int64_t group_size) {
  const c10::cuda::CUDAGuard device_guard(hidden.device());
  const auto stream = at::cuda::getCurrentCUDAStream(hidden.get_device());
  const std::int64_t rows = hidden.size(0);
  auto intermediate = torch::empty(
      {rows * kTopK, kIntermediateSize},
      hidden.options().dtype(torch::kBFloat16));
  auto output =
      torch::empty({rows, kHiddenSize}, hidden.options().dtype(torch::kBFloat16));

  const dim3 gate_up_grid(
      kIntermediateSize, static_cast<unsigned int>(rows * kTopK), 1);
  packed_q8_gate_up_swiglu_kernel<<<gate_up_grid, kThreads, 0, stream.stream()>>>(
      hidden.data_ptr<c10::BFloat16>(),
      expert_indices.data_ptr<std::int64_t>(),
      gate_quantized.data_ptr<std::int8_t>(),
      gate_scales.data_ptr<float>(),
      up_quantized.data_ptr<std::int8_t>(),
      up_scales.data_ptr<float>(),
      intermediate.data_ptr<c10::BFloat16>(),
      rows,
      group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 down_grid(kHiddenSize, static_cast<unsigned int>(rows), 1);
  packed_q8_down_route_kernel<<<down_grid, kThreads, 0, stream.stream()>>>(
      intermediate.data_ptr<c10::BFloat16>(),
      expert_indices.data_ptr<std::int64_t>(),
      routing_weights.data_ptr<c10::BFloat16>(),
      down_quantized.data_ptr<std::int8_t>(),
      down_scales.data_ptr<float>(),
      output.data_ptr<c10::BFloat16>(),
      rows,
      group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor uma_qmoe_q8_moe_prefill_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& expert_indices,
    const torch::Tensor& routing_weights,
    const torch::Tensor& route_order,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& gate_quantized,
    const torch::Tensor& gate_scales,
    const torch::Tensor& up_quantized,
    const torch::Tensor& up_scales,
    const torch::Tensor& down_quantized,
    const torch::Tensor& down_scales,
    std::int64_t group_size) {
  const c10::cuda::CUDAGuard device_guard(hidden.device());
  const auto stream = at::cuda::getCurrentCUDAStream(hidden.get_device());
  const std::int64_t rows = hidden.size(0);
  const std::int64_t route_count = rows * kTopK;
  auto intermediate = torch::empty(
      {route_count, kIntermediateSize},
      hidden.options().dtype(torch::kBFloat16));
  auto route_output = torch::empty(
      {route_count, kHiddenSize}, hidden.options().dtype(torch::kBFloat16));
  auto output =
      torch::empty({rows, kHiddenSize}, hidden.options().dtype(torch::kBFloat16));

  const unsigned int compact_route_tiles = static_cast<unsigned int>(
      (route_count + kRouteTile - 1) / kRouteTile + kExpertCount - 1);
  const dim3 tiled_block(kOutputTile, kRouteTile, 1);
  const dim3 gate_up_grid(
      (kIntermediateSize + kVectorOutputTile - 1) / kVectorOutputTile,
      compact_route_tiles,
      1);
  packed_q8_gate_up_vectorized_kernel<<<
      gate_up_grid, tiled_block, 0, stream.stream()>>>(
      hidden.data_ptr<c10::BFloat16>(),
      route_order.data_ptr<std::int64_t>(),
      expert_offsets.data_ptr<std::int64_t>(),
      gate_quantized.data_ptr<std::int8_t>(),
      gate_scales.data_ptr<float>(),
      up_quantized.data_ptr<std::int8_t>(),
      up_scales.data_ptr<float>(),
      intermediate.data_ptr<c10::BFloat16>(),
      group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 down_grid(
      (kHiddenSize + kVectorOutputTile - 1) / kVectorOutputTile,
      compact_route_tiles,
      1);
  packed_q8_down_vectorized_kernel<<<
      down_grid, tiled_block, 0, stream.stream()>>>(
      intermediate.data_ptr<c10::BFloat16>(),
      routing_weights.data_ptr<c10::BFloat16>(),
      route_order.data_ptr<std::int64_t>(),
      expert_offsets.data_ptr<std::int64_t>(),
      down_quantized.data_ptr<std::int8_t>(),
      down_scales.data_ptr<float>(),
      route_output.data_ptr<c10::BFloat16>(),
      group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 reduce_grid(
      (kHiddenSize + kThreads - 1) / kThreads,
      static_cast<unsigned int>(rows),
      1);
  reduce_topk_route_output_kernel<<<reduce_grid, kThreads, 0, stream.stream()>>>(
      route_output.data_ptr<c10::BFloat16>(),
      output.data_ptr<c10::BFloat16>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
