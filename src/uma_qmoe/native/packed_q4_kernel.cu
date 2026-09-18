#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;

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
  for (std::int64_t column = threadIdx.x; column < input_features;
       column += blockDim.x) {
    const std::int64_t weight_index = weight_row + column;
    const std::uint8_t byte = packed[weight_index >> 1];
    const std::uint8_t nibble =
        (weight_index & 1) == 0 ? (byte & 0x0f) : (byte >> 4);
    const int quantized = nibble >= 8 ? static_cast<int>(nibble) - 16
                                      : static_cast<int>(nibble);
    partial += static_cast<float>(input[input_row + column]) *
               (static_cast<float>(quantized) * scales[weight_index / group_size]);
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
