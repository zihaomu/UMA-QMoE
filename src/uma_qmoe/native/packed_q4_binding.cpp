#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr std::int64_t kHiddenSize = 2048;
constexpr std::int64_t kIntermediateSize = 1024;
constexpr std::int64_t kExpertCount = 64;
constexpr std::int64_t kTopK = 8;

void check_device_tensor(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name,
    torch::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on a CUDA/HIP device");
  TORCH_CHECK(
      tensor.device() == reference.device(), name, " must share the input device");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unexpected dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_projection_storage(
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    std::int64_t output_features,
    std::int64_t input_features,
    std::int64_t group_size,
    const char* name) {
  const auto elements = kExpertCount * output_features * input_features;
  TORCH_CHECK(
      packed.dim() == 1 && packed.numel() == elements / 2,
      name,
      " packed byte count is invalid");
  TORCH_CHECK(
      scales.dim() == 1 && scales.numel() == elements / group_size,
      name,
      " scale count is invalid");
}

}  // namespace

torch::Tensor uma_qmoe_q4_linear_cuda(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    std::int64_t output_features,
    std::int64_t input_features,
    std::int64_t group_size);

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
    std::int64_t group_size);

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
    std::int64_t group_size);

torch::Tensor q4_linear(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    std::int64_t output_features,
    std::int64_t input_features,
    std::int64_t group_size) {
  TORCH_CHECK(input.is_cuda(), "packed Q4 input must be on a CUDA/HIP device");
  check_device_tensor(packed, input, "packed Q4 bytes", torch::kUInt8);
  check_device_tensor(scales, input, "packed Q4 scales", torch::kFloat32);
  TORCH_CHECK(
      input.scalar_type() == torch::kBFloat16,
      "packed Q4 v1 accepts BF16 activations only");
  TORCH_CHECK(input.dim() == 2, "packed Q4 input must have shape [rows, K]");
  TORCH_CHECK(input.is_contiguous(), "packed Q4 input must be contiguous");
  TORCH_CHECK(
      output_features > 0 && input_features > 0 && group_size > 0,
      "packed Q4 dimensions and group size must be positive");
  TORCH_CHECK(
      input_features % 2 == 0 && group_size % 2 == 0,
      "packed Q4 byte-pair kernel requires even K and group size");
  TORCH_CHECK(
      input.size(1) == input_features,
      "packed Q4 input K does not match the weight shape");
  const auto element_count = output_features * input_features;
  TORCH_CHECK(
      packed.dim() == 1 && packed.numel() == element_count / 2,
      "packed Q4 byte count does not match the weight shape");
  TORCH_CHECK(
      scales.dim() == 1 && scales.numel() == element_count / group_size,
      "packed Q4 scale count does not match the weight shape");
  return uma_qmoe_q4_linear_cuda(
      input, packed, scales, output_features, input_features, group_size);
}

torch::Tensor q4_moe_forward(
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
  TORCH_CHECK(hidden.is_cuda(), "fused Q4 MoE input must be on a CUDA/HIP device");
  TORCH_CHECK(
      hidden.scalar_type() == torch::kBFloat16,
      "fused Q4 MoE accepts BF16 activations only");
  TORCH_CHECK(
      hidden.dim() == 2 && hidden.size(1) == kHiddenSize,
      "fused Q4 MoE input must have shape [tokens, 2048]");
  TORCH_CHECK(hidden.is_contiguous(), "fused Q4 MoE input must be contiguous");
  check_device_tensor(expert_indices, hidden, "expert indices", torch::kInt64);
  check_device_tensor(routing_weights, hidden, "routing weights", torch::kBFloat16);
  TORCH_CHECK(
      expert_indices.dim() == 2 && expert_indices.size(0) == hidden.size(0) &&
          expert_indices.size(1) == kTopK,
      "expert indices must have shape [tokens, 8]");
  TORCH_CHECK(
      routing_weights.sizes() == expert_indices.sizes(),
      "routing weights must match expert indices");
  TORCH_CHECK(group_size == 128, "fused Q4 MoE requires group size 128");

  check_device_tensor(gate_packed, hidden, "gate packed", torch::kUInt8);
  check_device_tensor(up_packed, hidden, "up packed", torch::kUInt8);
  check_device_tensor(down_packed, hidden, "down packed", torch::kUInt8);
  check_device_tensor(gate_scales, hidden, "gate scales", torch::kFloat32);
  check_device_tensor(up_scales, hidden, "up scales", torch::kFloat32);
  check_device_tensor(down_scales, hidden, "down scales", torch::kFloat32);
  check_projection_storage(
      gate_packed,
      gate_scales,
      kIntermediateSize,
      kHiddenSize,
      group_size,
      "gate");
  check_projection_storage(
      up_packed,
      up_scales,
      kIntermediateSize,
      kHiddenSize,
      group_size,
      "up");
  check_projection_storage(
      down_packed,
      down_scales,
      kHiddenSize,
      kIntermediateSize,
      group_size,
      "down");
  return uma_qmoe_q4_moe_forward_cuda(
      hidden,
      expert_indices,
      routing_weights,
      gate_packed,
      gate_scales,
      up_packed,
      up_scales,
      down_packed,
      down_scales,
      group_size);
}

torch::Tensor q4_moe_prefill(
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
  TORCH_CHECK(hidden.is_cuda(), "Q4 MoE prefill input must be on a CUDA/HIP device");
  TORCH_CHECK(
      hidden.scalar_type() == torch::kBFloat16,
      "Q4 MoE prefill accepts BF16 activations only");
  TORCH_CHECK(
      hidden.dim() == 2 && hidden.size(0) > 1 && hidden.size(1) == kHiddenSize,
      "Q4 MoE prefill input must have shape [tokens > 1, 2048]");
  TORCH_CHECK(hidden.is_contiguous(), "Q4 MoE prefill input must be contiguous");
  check_device_tensor(expert_indices, hidden, "expert indices", torch::kInt64);
  check_device_tensor(routing_weights, hidden, "routing weights", torch::kBFloat16);
  check_device_tensor(route_order, hidden, "route order", torch::kInt64);
  check_device_tensor(expert_offsets, hidden, "expert offsets", torch::kInt64);
  TORCH_CHECK(
      expert_indices.dim() == 2 && expert_indices.size(0) == hidden.size(0) &&
          expert_indices.size(1) == kTopK,
      "expert indices must have shape [tokens, 8]");
  TORCH_CHECK(
      routing_weights.sizes() == expert_indices.sizes(),
      "routing weights must match expert indices");
  TORCH_CHECK(
      route_order.dim() == 1 && route_order.numel() == expert_indices.numel(),
      "route order must contain every flattened route");
  TORCH_CHECK(
      expert_offsets.dim() == 1 && expert_offsets.numel() == kExpertCount + 1,
      "expert offsets must have shape [65]");
  TORCH_CHECK(group_size == 128, "Q4 MoE prefill requires group size 128");

  check_device_tensor(gate_packed, hidden, "gate packed", torch::kUInt8);
  check_device_tensor(up_packed, hidden, "up packed", torch::kUInt8);
  check_device_tensor(down_packed, hidden, "down packed", torch::kUInt8);
  check_device_tensor(gate_scales, hidden, "gate scales", torch::kFloat32);
  check_device_tensor(up_scales, hidden, "up scales", torch::kFloat32);
  check_device_tensor(down_scales, hidden, "down scales", torch::kFloat32);
  check_projection_storage(
      gate_packed,
      gate_scales,
      kIntermediateSize,
      kHiddenSize,
      group_size,
      "gate");
  check_projection_storage(
      up_packed,
      up_scales,
      kIntermediateSize,
      kHiddenSize,
      group_size,
      "up");
  check_projection_storage(
      down_packed,
      down_scales,
      kHiddenSize,
      kIntermediateSize,
      group_size,
      "down");
  return uma_qmoe_q4_moe_prefill_cuda(
      hidden,
      expert_indices,
      routing_weights,
      route_order,
      expert_offsets,
      gate_packed,
      gate_scales,
      up_packed,
      up_scales,
      down_packed,
      down_scales,
      group_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "q4_linear",
      &q4_linear,
      "UMA-QMoE canonical packed-Q4 W4A16 linear (CUDA/HIP)");
  module.def(
      "q4_moe_forward",
      &q4_moe_forward,
      "UMA-QMoE fused Gate+Up+SwiGLU+Down Top-8 packed-Q4 MoE (CUDA/HIP)");
  module.def(
      "q4_moe_prefill",
      &q4_moe_prefill,
      "UMA-QMoE expert-sorted packed-Q4 MoE prefill (CUDA/HIP)");
}
