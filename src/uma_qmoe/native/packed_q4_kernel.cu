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
    const std::uint8_t* __restrict__ gate_packed,
    const float* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_packed,
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
