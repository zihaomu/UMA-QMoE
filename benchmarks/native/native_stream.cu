#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define GPU_BACKEND "hip"
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemset hipMemset
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventDestroy hipEventDestroy
#define GPU_LAUNCH(kernel, blocks, threads, ...) \
  hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), 0, 0, __VA_ARGS__)
#else
#include <cuda_runtime.h>
#define GPU_BACKEND "cuda"
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceProp_t cudaDeviceProp
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemset cudaMemset
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventDestroy cudaEventDestroy
#define GPU_LAUNCH(kernel, blocks, threads, ...) \
  kernel<<<blocks, threads>>>(__VA_ARGS__)
#endif

namespace {

bool check(gpuError_t status, const char* expression) {
  if (status == gpuSuccess) return true;
  std::cerr << expression << ": " << gpuGetErrorString(status) << "\n";
  return false;
}

std::string json_escape(const char* value) {
  std::ostringstream output;
  for (const unsigned char character : std::string(value)) {
    if (character == '\\' || character == '"') output << '\\';
    if (character >= 0x20) output << character;
  }
  return output.str();
}

__global__ void write_kernel(float4* destination, std::size_t count, float value) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  const float4 item = make_float4(value, value, value, value);
  for (std::size_t position = index; position < count; position += stride) {
    destination[position] = item;
  }
}

__global__ void copy_kernel(const float4* source, float4* destination, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t position = index; position < count; position += stride) {
    destination[position] = source[position];
  }
}

__global__ void read_reduce_kernel(const float4* source, std::size_t count, float* sink) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  float sum = 0.0f;
  for (std::size_t position = index; position < count; position += stride) {
    const float4 item = source[position];
    sum += item.x + item.y + item.z + item.w;
  }
  atomicAdd(sink, sum);
}

template <typename Launch>
bool measure(
    Launch launch,
    int warmup,
    int iterations,
    int inner,
    std::vector<double>* samples) {
  for (int sample = 0; sample < warmup; ++sample) {
    for (int loop = 0; loop < inner; ++loop) launch(loop);
    if (!check(gpuDeviceSynchronize(), "gpuDeviceSynchronize(warmup)")) return false;
  }
  gpuEvent_t start{};
  gpuEvent_t stop{};
  if (!check(gpuEventCreate(&start), "gpuEventCreate(start)")) return false;
  if (!check(gpuEventCreate(&stop), "gpuEventCreate(stop)")) return false;
  for (int sample = 0; sample < iterations; ++sample) {
    if (!check(gpuEventRecord(start), "gpuEventRecord(start)")) return false;
    for (int loop = 0; loop < inner; ++loop) launch(loop);
    if (!check(gpuEventRecord(stop), "gpuEventRecord(stop)")) return false;
    if (!check(gpuEventSynchronize(stop), "gpuEventSynchronize(stop)")) return false;
    float milliseconds = 0.0f;
    if (!check(gpuEventElapsedTime(&milliseconds, start, stop), "gpuEventElapsedTime")) {
      return false;
    }
    samples->push_back(static_cast<double>(milliseconds) / 1000.0);
  }
  check(gpuEventDestroy(start), "gpuEventDestroy(start)");
  check(gpuEventDestroy(stop), "gpuEventDestroy(stop)");
  return true;
}

void print_operation(const char* id, const std::vector<double>& samples, bool comma) {
  if (comma) std::cout << ',';
  std::cout << "{\"id\":\"" << id << "\",\"samples_seconds\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << std::setprecision(12) << samples[index];
  }
  std::cout << "]}";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: native-stream TARGET_ID BYTES WARMUP ITERATIONS INNER\n";
    return 2;
  }
  const std::size_t requested_bytes = std::strtoull(argv[2], nullptr, 10);
  const int warmup = std::atoi(argv[3]);
  const int iterations = std::atoi(argv[4]);
  const int inner = std::atoi(argv[5]);
  const std::size_t vector_count = requested_bytes / sizeof(float4);
  const std::size_t actual_bytes = vector_count * sizeof(float4);
  if (vector_count == 0 || warmup < 1 || iterations < 3 || inner < 1) return 2;

  int device = 0;
  gpuDeviceProp_t properties{};
  if (!check(gpuGetDevice(&device), "gpuGetDevice")) return 3;
  if (!check(gpuGetDeviceProperties(&properties, device), "gpuGetDeviceProperties")) return 3;
  const int threads = 256;
  const int blocks = properties.multiProcessorCount * 32;

  float4* source = nullptr;
  float4* destination = nullptr;
  float* sink = nullptr;
  if (!check(gpuMalloc(reinterpret_cast<void**>(&source), actual_bytes), "gpuMalloc(source)")) return 4;
  if (!check(gpuMalloc(reinterpret_cast<void**>(&destination), actual_bytes), "gpuMalloc(destination)")) return 4;
  if (!check(gpuMalloc(reinterpret_cast<void**>(&sink), sizeof(float)), "gpuMalloc(sink)")) return 4;
  GPU_LAUNCH(write_kernel, blocks, threads, source, vector_count, 1.0f);
  if (!check(gpuDeviceSynchronize(), "gpuDeviceSynchronize(initialize)")) return 5;

  std::vector<double> read_samples;
  std::vector<double> write_samples;
  std::vector<double> copy_samples;
  auto read_launch = [&](int) {
    gpuMemset(sink, 0, sizeof(float));
    GPU_LAUNCH(read_reduce_kernel, blocks, threads, source, vector_count, sink);
  };
  auto write_launch = [&](int loop) {
    GPU_LAUNCH(write_kernel, blocks, threads, destination, vector_count, (loop & 1) ? 1.0f : 2.0f);
  };
  auto copy_launch = [&](int) {
    GPU_LAUNCH(copy_kernel, blocks, threads, source, destination, vector_count);
  };
  if (!measure(read_launch, warmup, iterations, inner, &read_samples)) return 5;
  if (!measure(write_launch, warmup, iterations, inner, &write_samples)) return 5;
  if (!measure(copy_launch, warmup, iterations, inner, &copy_samples)) return 5;

  std::cout << "{\"backend\":\"" << GPU_BACKEND << "\","
            << "\"device_name\":\"" << json_escape(properties.name) << "\","
            << "\"multiprocessor_count\":" << properties.multiProcessorCount << ','
            << "\"threads_per_block\":" << threads << ','
            << "\"blocks\":" << blocks << ','
            << "\"actual_buffer_bytes\":" << actual_bytes << ",\"operations\":[";
  print_operation("read_reduce", read_samples, false);
  print_operation("write", write_samples, true);
  print_operation("copy", copy_samples, true);
  std::cout << "]}\n";

  check(gpuFree(sink), "gpuFree(sink)");
  check(gpuFree(destination), "gpuFree(destination)");
  check(gpuFree(source), "gpuFree(source)");
  return 0;
}
