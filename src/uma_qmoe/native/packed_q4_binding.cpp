#include <torch/extension.h>

#include <cstdint>

torch::Tensor uma_qmoe_q4_linear_cuda(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    std::int64_t output_features,
    std::int64_t input_features,
    std::int64_t group_size);

torch::Tensor q4_linear(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    std::int64_t output_features,
    std::int64_t input_features,
    std::int64_t group_size) {
  TORCH_CHECK(input.is_cuda(), "packed Q4 input must be on a CUDA/HIP device");
  TORCH_CHECK(packed.is_cuda(), "packed Q4 bytes must be on a CUDA/HIP device");
  TORCH_CHECK(scales.is_cuda(), "packed Q4 scales must be on a CUDA/HIP device");
  TORCH_CHECK(input.device() == packed.device() && input.device() == scales.device(),
              "packed Q4 operands must be on one device");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "packed Q4 v1 accepts BF16 activations only");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8,
              "packed Q4 weights must use uint8 storage");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "packed Q4 scales must use float32 storage");
  TORCH_CHECK(input.dim() == 2, "packed Q4 input must have shape [rows, K]");
  TORCH_CHECK(packed.dim() == 1 && scales.dim() == 1,
              "packed Q4 storage tensors must be flat");
  TORCH_CHECK(input.is_contiguous() && packed.is_contiguous() && scales.is_contiguous(),
              "packed Q4 operands must be contiguous");
  TORCH_CHECK(output_features > 0 && input_features > 0 && group_size > 0,
              "packed Q4 dimensions and group size must be positive");
  TORCH_CHECK(input.size(1) == input_features,
              "packed Q4 input K does not match the weight shape");
  const auto element_count = output_features * input_features;
  TORCH_CHECK(packed.numel() == (element_count + 1) / 2,
              "packed Q4 byte count does not match the weight shape");
  TORCH_CHECK(scales.numel() == (element_count + group_size - 1) / group_size,
              "packed Q4 scale count does not match the weight shape");
  return uma_qmoe_q4_linear_cuda(
      input, packed, scales, output_features, input_features, group_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "q4_linear",
      &q4_linear,
      "UMA-QMoE canonical packed-Q4 W4A16 linear (CUDA/HIP)");
}
