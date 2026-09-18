#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define GPU_BACKEND "hip"
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuMallocManaged hipMallocManaged
#define gpuFree hipFree
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetLastError hipGetLastError
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventDestroy hipEventDestroy
#define gpuMemAddressReserve hipMemAddressReserve
#define gpuMemCreate hipMemCreate
#define gpuMemMap hipMemMap
#define gpuMemSetAccess hipMemSetAccess
#define gpuMemUnmap hipMemUnmap
#define gpuMemRelease hipMemRelease
#define gpuMemAddressFree hipMemAddressFree
#define gpuMemGetAllocationGranularity hipMemGetAllocationGranularity
#define gpuMemAllocationProp hipMemAllocationProp
#define gpuMemAccessDesc hipMemAccessDesc
#define gpuMemGenericAllocationHandle_t hipMemGenericAllocationHandle_t
#define gpuMemAllocationTypePinned hipMemAllocationTypePinned
#define gpuMemLocationTypeDevice hipMemLocationTypeDevice
#define gpuMemAccessFlagsProtReadWrite hipMemAccessFlagsProtReadWrite
#define gpuMemAllocationGranularityMinimum hipMemAllocationGranularityMinimum
#define GPU_MANAGED_ATTRIBUTE hipDeviceAttributeManagedMemory
#define GPU_CONCURRENT_MANAGED_ATTRIBUTE hipDeviceAttributeConcurrentManagedAccess
#define GPU_PAGEABLE_ATTRIBUTE hipDeviceAttributePageableMemoryAccess
#define GPU_PAGE_TABLE_ATTRIBUTE hipDeviceAttributePageableMemoryAccessUsesHostPageTables
#define GPU_LAUNCH(kernel, blocks, threads, ...) \
  hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), 0, 0, __VA_ARGS__)
#else
#include <cuda.h>
#include <cuda_runtime.h>
#define GPU_BACKEND "cuda"
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#define gpuDeviceProp_t cudaDeviceProp
#define gpuMallocManaged cudaMallocManaged
#define gpuFree cudaFree
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventDestroy cudaEventDestroy
#define gpuMemAllocationProp CUmemAllocationProp
#define gpuMemAccessDesc CUmemAccessDesc
#define gpuMemGenericAllocationHandle_t CUmemGenericAllocationHandle
#define gpuMemAllocationTypePinned CU_MEM_ALLOCATION_TYPE_PINNED
#define gpuMemLocationTypeDevice CU_MEM_LOCATION_TYPE_DEVICE
#define gpuMemAccessFlagsProtReadWrite CU_MEM_ACCESS_FLAGS_PROT_READWRITE
#define gpuMemAllocationGranularityMinimum CU_MEM_ALLOC_GRANULARITY_MINIMUM
#define GPU_MANAGED_ATTRIBUTE cudaDevAttrManagedMemory
#define GPU_CONCURRENT_MANAGED_ATTRIBUTE cudaDevAttrConcurrentManagedAccess
#define GPU_PAGEABLE_ATTRIBUTE cudaDevAttrPageableMemoryAccess
#define GPU_PAGE_TABLE_ATTRIBUTE cudaDevAttrPageableMemoryAccessUsesHostPageTables
#define GPU_LAUNCH(kernel, blocks, threads, ...) kernel<<<blocks, threads>>>(__VA_ARGS__)
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Faults {
  long minor = 0;
  long major = 0;
};

struct TouchResult {
  std::string id;
  std::string status = "unavailable";
  std::string reason;
  double seconds = 0.0;
  long minor_faults_delta = 0;
  long major_faults_delta = 0;
  std::string following_actor;
  std::string following_status = "unavailable";
  std::string following_reason;
  double following_seconds = 0.0;
};

struct OperationResult {
  std::string id;
  std::vector<double> samples_seconds;
};

struct ContentionResult {
  std::string status = "unavailable";
  std::string reason;
  std::vector<double> wall_samples_seconds;
  std::vector<double> gpu_samples_seconds;
};

struct CaseResult {
  std::string id;
  std::string allocation_api;
  std::string access_path;
  std::string status = "unavailable";
  std::string reason;
  std::size_t buffer_bytes = 0;
  double allocation_seconds = 0.0;
  std::vector<TouchResult> touches;
  std::vector<OperationResult> operations;
  ContentionResult contention;
};

struct Allocation {
  float* source = nullptr;
  float* destination = nullptr;
  std::size_t bytes = 0;
  bool host_accessible = false;
  bool contention_supported = false;
  void (*release)(Allocation*) = nullptr;
  std::uintptr_t source_address = 0;
  std::uintptr_t destination_address = 0;
  gpuMemGenericAllocationHandle_t source_handle{};
  gpuMemGenericAllocationHandle_t destination_handle{};
};

bool check(gpuError_t status, const char* expression) {
  if (status == gpuSuccess) return true;
  std::cerr << expression << ": " << gpuGetErrorString(status) << "\n";
  return false;
}

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
bool driver_check(CUresult status, const char* expression) {
  if (status == CUDA_SUCCESS) return true;
  const char* message = "unknown CUDA driver error";
  cuGetErrorString(status, &message);
  std::cerr << expression << ": " << message << "\n";
  return false;
}
#endif

bool vmm_initialize() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return true;
#else
  return driver_check(cuInit(0), "cuInit");
#endif
}

bool vmm_get_granularity(
    std::size_t* granularity, const gpuMemAllocationProp* property) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return check(
      gpuMemGetAllocationGranularity(
          granularity, property, gpuMemAllocationGranularityMinimum),
      "hipMemGetAllocationGranularity");
#else
  return driver_check(
      cuMemGetAllocationGranularity(
          granularity, property, gpuMemAllocationGranularityMinimum),
      "cuMemGetAllocationGranularity");
#endif
}

bool vmm_create(
    gpuMemGenericAllocationHandle_t* handle,
    std::size_t bytes,
    const gpuMemAllocationProp* property) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return check(gpuMemCreate(handle, bytes, property, 0), "hipMemCreate");
#else
  return driver_check(cuMemCreate(handle, bytes, property, 0), "cuMemCreate");
#endif
}

bool vmm_reserve(std::uintptr_t* address, std::size_t bytes) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  void* reserved = nullptr;
  if (!check(
          gpuMemAddressReserve(&reserved, bytes, 0, nullptr, 0),
          "hipMemAddressReserve")) {
    return false;
  }
  *address = reinterpret_cast<std::uintptr_t>(reserved);
  return true;
#else
  CUdeviceptr reserved = 0;
  if (!driver_check(
          cuMemAddressReserve(&reserved, bytes, 0, 0, 0),
          "cuMemAddressReserve")) {
    return false;
  }
  *address = static_cast<std::uintptr_t>(reserved);
  return true;
#endif
}

bool vmm_map(
    std::uintptr_t address,
    std::size_t bytes,
    gpuMemGenericAllocationHandle_t handle) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return check(
      gpuMemMap(reinterpret_cast<void*>(address), bytes, 0, handle, 0),
      "hipMemMap");
#else
  return driver_check(
      cuMemMap(static_cast<CUdeviceptr>(address), bytes, 0, handle, 0),
      "cuMemMap");
#endif
}

bool vmm_set_access(
    std::uintptr_t address,
    std::size_t bytes,
    const gpuMemAccessDesc* access) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return check(
      gpuMemSetAccess(reinterpret_cast<void*>(address), bytes, access, 1),
      "hipMemSetAccess");
#else
  return driver_check(
      cuMemSetAccess(static_cast<CUdeviceptr>(address), bytes, access, 1),
      "cuMemSetAccess");
#endif
}

bool vmm_unmap(std::uintptr_t address, std::size_t bytes) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return check(
      gpuMemUnmap(reinterpret_cast<void*>(address), bytes), "hipMemUnmap");
#else
  return driver_check(
      cuMemUnmap(static_cast<CUdeviceptr>(address), bytes), "cuMemUnmap");
#endif
}

bool vmm_address_free(std::uintptr_t address, std::size_t bytes) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return check(
      gpuMemAddressFree(reinterpret_cast<void*>(address), bytes),
      "hipMemAddressFree");
#else
  return driver_check(
      cuMemAddressFree(static_cast<CUdeviceptr>(address), bytes),
      "cuMemAddressFree");
#endif
}

bool vmm_release(gpuMemGenericAllocationHandle_t handle) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return check(gpuMemRelease(handle), "hipMemRelease");
#else
  return driver_check(cuMemRelease(handle), "cuMemRelease");
#endif
}

std::string json_escape(const char* value) {
  std::ostringstream output;
  for (const unsigned char character : std::string(value)) {
    if (character == '\\' || character == '"') output << '\\';
    if (character >= 0x20) output << character;
  }
  return output.str();
}

Faults faults() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return {usage.ru_minflt, usage.ru_majflt};
}

double elapsed_seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

__global__ void write_kernel(float* destination, std::size_t count, float value) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t position = index; position < count; position += stride) {
    destination[position] = value;
  }
}

__global__ void copy_kernel(
    const float* source, float* destination, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t position = index; position < count; position += stride) {
    destination[position] = source[position];
  }
}

__global__ void read_kernel(const float* source, std::size_t count, float* sink) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  float sum = 0.0f;
  for (std::size_t position = index; position < count; position += stride) {
    sum += source[position];
  }
  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) sink[blockIdx.x] = reduction[0];
}

template <typename Launch>
bool gpu_elapsed(Launch launch, double* seconds) {
  gpuEvent_t start{};
  gpuEvent_t stop{};
  if (!check(gpuEventCreate(&start), "gpuEventCreate(start)")) return false;
  if (!check(gpuEventCreate(&stop), "gpuEventCreate(stop)")) {
    static_cast<void>(gpuEventDestroy(start));
    return false;
  }
  bool ok = check(gpuEventRecord(start), "gpuEventRecord(start)");
  if (ok) launch();
  if (ok) ok = check(gpuGetLastError(), "kernel launch");
  if (ok) ok = check(gpuEventRecord(stop), "gpuEventRecord(stop)");
  if (ok) ok = check(gpuEventSynchronize(stop), "gpuEventSynchronize(stop)");
  float milliseconds = 0.0f;
  if (ok) ok = check(
      gpuEventElapsedTime(&milliseconds, start, stop), "gpuEventElapsedTime");
  static_cast<void>(gpuEventDestroy(start));
  static_cast<void>(gpuEventDestroy(stop));
  if (!ok || !std::isfinite(milliseconds) || milliseconds <= 0.0f) return false;
  *seconds = static_cast<double>(milliseconds) / 1000.0;
  return true;
}

template <typename Launch>
bool measure_gpu(
    Launch launch,
    int warmup,
    int iterations,
    int inner,
    std::vector<double>* samples) {
  for (int sample = 0; sample < warmup; ++sample) {
    for (int loop = 0; loop < inner; ++loop) launch(loop);
    if (!check(gpuGetLastError(), "warmup kernel launch")) return false;
    if (!check(gpuDeviceSynchronize(), "gpuDeviceSynchronize(warmup)")) return false;
  }
  for (int sample = 0; sample < iterations; ++sample) {
    double seconds = 0.0;
    if (!gpu_elapsed(
            [&]() {
              for (int loop = 0; loop < inner; ++loop) launch(loop);
            },
            &seconds)) {
      return false;
    }
    samples->push_back(seconds);
  }
  return true;
}

void release_managed(Allocation* allocation) {
  if (allocation->destination != nullptr) {
    static_cast<void>(gpuFree(allocation->destination));
  }
  if (allocation->source != nullptr) {
    static_cast<void>(gpuFree(allocation->source));
  }
  allocation->source = nullptr;
  allocation->destination = nullptr;
}

void release_pageable(Allocation* allocation) {
  std::free(allocation->destination);
  std::free(allocation->source);
  allocation->source = nullptr;
  allocation->destination = nullptr;
}

void release_vmm(Allocation* allocation) {
  if (allocation->destination_address != 0) {
    static_cast<void>(
        vmm_unmap(allocation->destination_address, allocation->bytes));
    static_cast<void>(
        vmm_address_free(allocation->destination_address, allocation->bytes));
    static_cast<void>(vmm_release(allocation->destination_handle));
  }
  if (allocation->source_address != 0) {
    static_cast<void>(vmm_unmap(allocation->source_address, allocation->bytes));
    static_cast<void>(
        vmm_address_free(allocation->source_address, allocation->bytes));
    static_cast<void>(vmm_release(allocation->source_handle));
  }
  allocation->source = nullptr;
  allocation->destination = nullptr;
}

bool allocate_managed(std::size_t bytes, bool concurrent, Allocation* output) {
  output->bytes = bytes;
  output->host_accessible = true;
  output->contention_supported = concurrent;
  output->release = release_managed;
  if (!check(
          gpuMallocManaged(reinterpret_cast<void**>(&output->source), bytes),
          "gpuMallocManaged(source)")) {
    return false;
  }
  if (!check(
          gpuMallocManaged(reinterpret_cast<void**>(&output->destination), bytes),
          "gpuMallocManaged(destination)")) {
    release_managed(output);
    return false;
  }
  return true;
}

bool allocate_pageable(std::size_t bytes, Allocation* output) {
  output->bytes = bytes;
  output->host_accessible = true;
  output->contention_supported = true;
  output->release = release_pageable;
  if (posix_memalign(reinterpret_cast<void**>(&output->source), 4096, bytes) != 0) {
    return false;
  }
  if (posix_memalign(
          reinterpret_cast<void**>(&output->destination), 4096, bytes) != 0) {
    release_pageable(output);
    return false;
  }
  return true;
}

bool allocate_one_vmm(
    int device,
    std::size_t bytes,
    std::uintptr_t* address,
    gpuMemGenericAllocationHandle_t* handle) {
  gpuMemAllocationProp property{};
  property.type = gpuMemAllocationTypePinned;
  property.location.type = gpuMemLocationTypeDevice;
  property.location.id = device;
  if (!vmm_create(handle, bytes, &property)) return false;
  if (!vmm_reserve(address, bytes)) {
    static_cast<void>(vmm_release(*handle));
    return false;
  }
  if (!vmm_map(*address, bytes, *handle)) {
    static_cast<void>(vmm_address_free(*address, bytes));
    static_cast<void>(vmm_release(*handle));
    *address = 0;
    return false;
  }
  gpuMemAccessDesc access{};
  access.location.type = gpuMemLocationTypeDevice;
  access.location.id = device;
  access.flags = gpuMemAccessFlagsProtReadWrite;
  if (!vmm_set_access(*address, bytes, &access)) {
    static_cast<void>(vmm_unmap(*address, bytes));
    static_cast<void>(vmm_address_free(*address, bytes));
    static_cast<void>(vmm_release(*handle));
    *address = 0;
    return false;
  }
  return true;
}

bool allocate_vmm(int device, std::size_t requested_bytes, Allocation* output) {
  if (!vmm_initialize()) return false;
  gpuMemAllocationProp property{};
  property.type = gpuMemAllocationTypePinned;
  property.location.type = gpuMemLocationTypeDevice;
  property.location.id = device;
  std::size_t granularity = 0;
  if (!vmm_get_granularity(&granularity, &property)) {
    return false;
  }
  if (granularity == 0) return false;
  const std::size_t bytes =
      ((requested_bytes + granularity - 1) / granularity) * granularity;
  output->bytes = bytes;
  output->host_accessible = false;
  output->contention_supported = false;
  output->release = release_vmm;
  if (!allocate_one_vmm(
          device, bytes, &output->source_address, &output->source_handle)) {
    return false;
  }
  if (!allocate_one_vmm(
          device,
          bytes,
          &output->destination_address,
          &output->destination_handle)) {
    release_vmm(output);
    return false;
  }
  output->source = reinterpret_cast<float*>(output->source_address);
  output->destination = reinterpret_cast<float*>(output->destination_address);
  return true;
}

void host_write(float* pointer, std::size_t count, float value) {
  for (std::size_t index = 0; index < count; ++index) pointer[index] = value;
}

double host_sparse_read(const float* pointer, std::size_t count) {
  volatile float sink = 0.0f;
  constexpr std::size_t stride = 4096 / sizeof(float);
  const auto start = Clock::now();
  for (std::size_t index = 0; index < count; index += stride) sink += pointer[index];
  if (count > 0) sink += pointer[count - 1];
  static_cast<void>(sink);
  return elapsed_seconds(start);
}

CaseResult unavailable_case(
    const char* id,
    const char* allocation_api,
    const char* access_path,
    const char* reason) {
  CaseResult result;
  result.id = id;
  result.allocation_api = allocation_api;
  result.access_path = access_path;
  result.reason = reason;
  return result;
}

CaseResult measure_case(
    const char* id,
    const char* allocation_api,
    const char* access_path,
    Allocation* allocation,
    double allocation_seconds,
    int blocks,
    int threads,
    int warmup,
    int iterations,
    int inner) {
  CaseResult result;
  result.id = id;
  result.allocation_api = allocation_api;
  result.access_path = access_path;
  result.buffer_bytes = allocation->bytes;
  result.allocation_seconds = allocation_seconds;
  const std::size_t count = allocation->bytes / sizeof(float);

  float* sink = nullptr;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (!check(hipMalloc(reinterpret_cast<void**>(&sink),
                       static_cast<std::size_t>(blocks) * sizeof(float)),
             "hipMalloc(sink)")) {
#else
  if (!check(cudaMalloc(reinterpret_cast<void**>(&sink),
                        static_cast<std::size_t>(blocks) * sizeof(float)),
             "cudaMalloc(sink)")) {
#endif
    result.reason = "sink_allocation_failed";
    allocation->release(allocation);
    return result;
  }

  TouchResult cpu_touch;
  cpu_touch.id = "cpu_first";
  cpu_touch.following_actor = "gpu";
  if (allocation->host_accessible) {
    const Faults before = faults();
    const auto start = Clock::now();
    host_write(allocation->source, count, 1.0f);
    cpu_touch.seconds = elapsed_seconds(start);
    const Faults after = faults();
    cpu_touch.minor_faults_delta = after.minor - before.minor;
    cpu_touch.major_faults_delta = after.major - before.major;
    cpu_touch.status = "measured";
    if (gpu_elapsed(
            [&]() {
              GPU_LAUNCH(read_kernel, blocks, threads,
                         allocation->source, count, sink);
            },
            &cpu_touch.following_seconds)) {
      cpu_touch.following_status = "measured";
    } else {
      cpu_touch.following_reason = "following_gpu_access_failed";
    }
  } else {
    cpu_touch.reason = "allocation_not_host_accessible";
    cpu_touch.following_reason = "cpu_first_touch_unavailable";
  }
  result.touches.push_back(cpu_touch);

  TouchResult gpu_touch;
  gpu_touch.id = "gpu_first";
  gpu_touch.following_actor = "cpu";
  if (gpu_elapsed(
          [&]() {
            GPU_LAUNCH(write_kernel, blocks, threads,
                       allocation->destination, count, 2.0f);
          },
          &gpu_touch.seconds)) {
    gpu_touch.status = "measured";
    if (allocation->host_accessible) {
      const Faults before = faults();
      gpu_touch.following_seconds = host_sparse_read(allocation->destination, count);
      const Faults after = faults();
      gpu_touch.minor_faults_delta = after.minor - before.minor;
      gpu_touch.major_faults_delta = after.major - before.major;
      gpu_touch.following_status = "measured";
    } else {
      gpu_touch.following_reason = "allocation_not_host_accessible";
    }
  } else {
    gpu_touch.reason = "gpu_first_touch_failed";
    gpu_touch.following_reason = "gpu_first_touch_failed";
  }
  result.touches.push_back(gpu_touch);

  OperationResult read{"read", {}};
  OperationResult write{"write", {}};
  OperationResult copy{"copy", {}};
  const bool operations_ok =
      measure_gpu(
          [&](int) {
            GPU_LAUNCH(read_kernel, blocks, threads,
                       allocation->source, count, sink);
          },
          warmup, iterations, inner, &read.samples_seconds) &&
      measure_gpu(
          [&](int loop) {
            GPU_LAUNCH(write_kernel, blocks, threads,
                       allocation->destination, count,
                       (loop & 1) ? 1.0f : 2.0f);
          },
          warmup, iterations, inner, &write.samples_seconds) &&
      measure_gpu(
          [&](int) {
            GPU_LAUNCH(copy_kernel, blocks, threads,
                       allocation->source, allocation->destination, count);
          },
          warmup, iterations, inner, &copy.samples_seconds);
  if (!operations_ok) {
    result.reason = "steady_state_operation_failed";
  } else {
    result.operations = {read, write, copy};
    result.status = "measured";
  }

  if (result.status == "measured" && allocation->contention_supported) {
    const std::size_t half = count / 2;
    bool contention_ok = true;
    for (int sample = 0; sample < iterations; ++sample) {
      std::atomic<bool> start_cpu{false};
      std::thread cpu([&]() {
        while (!start_cpu.load(std::memory_order_acquire)) std::this_thread::yield();
        host_write(allocation->destination, half, static_cast<float>(sample & 1));
      });
      const auto wall_start = Clock::now();
      double gpu_seconds = 0.0;
      const bool sample_ok = gpu_elapsed(
          [&]() {
            start_cpu.store(true, std::memory_order_release);
            GPU_LAUNCH(write_kernel, blocks, threads,
                       allocation->destination + half, count - half, 3.0f);
          },
          &gpu_seconds);
      cpu.join();
      const double wall_seconds = elapsed_seconds(wall_start);
      if (!sample_ok || !std::isfinite(wall_seconds) || wall_seconds <= 0.0) {
        contention_ok = false;
        break;
      }
      result.contention.gpu_samples_seconds.push_back(gpu_seconds);
      result.contention.wall_samples_seconds.push_back(wall_seconds);
    }
    if (contention_ok) {
      result.contention.status = "measured";
    } else {
      result.contention.reason = "concurrent_access_failed";
      result.contention.gpu_samples_seconds.clear();
      result.contention.wall_samples_seconds.clear();
    }
  } else if (result.status == "measured") {
    result.contention.reason = "concurrent_host_device_access_not_supported";
  } else {
    result.contention.reason = "steady_state_measurement_failed";
  }

  static_cast<void>(gpuFree(sink));
  allocation->release(allocation);
  return result;
}

void print_samples(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << std::setprecision(12) << values[index];
  }
  std::cout << ']';
}

void print_touch(const TouchResult& touch) {
  std::cout << "{\"id\":\"" << touch.id << "\",\"status\":\""
            << touch.status << "\"";
  if (touch.status == "measured") {
    std::cout << ",\"seconds\":" << std::setprecision(12) << touch.seconds
              << ",\"minor_faults_delta\":" << touch.minor_faults_delta
              << ",\"major_faults_delta\":" << touch.major_faults_delta;
  } else {
    std::cout << ",\"reason\":\"" << touch.reason << "\"";
  }
  std::cout << ",\"following_access\":{\"actor\":\""
            << touch.following_actor << "\",\"status\":\""
            << touch.following_status << "\"";
  if (touch.following_status == "measured") {
    std::cout << ",\"seconds\":" << std::setprecision(12)
              << touch.following_seconds;
  } else {
    std::cout << ",\"reason\":\"" << touch.following_reason << "\"";
  }
  std::cout << "}}";
}

void print_case(const CaseResult& result) {
  std::cout << "{\"id\":\"" << result.id << "\",\"status\":\""
            << result.status << "\",\"allocation_api\":\""
            << result.allocation_api << "\",\"access_path\":\""
            << result.access_path << "\"";
  if (result.status == "unavailable") {
    std::cout << ",\"reason\":\"" << result.reason << "\"}";
    return;
  }
  std::cout << ",\"buffer_bytes\":" << result.buffer_bytes
            << ",\"allocation_seconds\":" << std::setprecision(12)
            << result.allocation_seconds << ",\"touches\":[";
  for (std::size_t index = 0; index < result.touches.size(); ++index) {
    if (index) std::cout << ',';
    print_touch(result.touches[index]);
  }
  std::cout << "],\"operations\":[";
  for (std::size_t index = 0; index < result.operations.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << "{\"id\":\"" << result.operations[index].id
              << "\",\"samples_seconds\":";
    print_samples(result.operations[index].samples_seconds);
    std::cout << '}';
  }
  std::cout << "],\"contention\":{\"status\":\""
            << result.contention.status << "\"";
  if (result.contention.status == "measured") {
    std::cout << ",\"wall_samples_seconds\":";
    print_samples(result.contention.wall_samples_seconds);
    std::cout << ",\"gpu_samples_seconds\":";
    print_samples(result.contention.gpu_samples_seconds);
  } else {
    std::cout << ",\"reason\":\"" << result.contention.reason << "\"";
  }
  std::cout << "}}";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: allocation-matrix-v2 TARGET_ID BYTES WARMUP ITERATIONS INNER\n";
    return 2;
  }
  char* end = nullptr;
  const unsigned long long parsed_bytes = std::strtoull(argv[2], &end, 10);
  if (end == argv[2] || *end != '\0') return 2;
  const std::size_t requested_bytes = static_cast<std::size_t>(parsed_bytes);
  const int warmup = std::atoi(argv[3]);
  const int iterations = std::atoi(argv[4]);
  const int inner = std::atoi(argv[5]);
  if (requested_bytes < 4 * 1024 * 1024 || requested_bytes % sizeof(float) != 0 ||
      warmup < 1 || iterations < 3 || inner < 1) {
    return 2;
  }

  int device = 0;
  gpuDeviceProp_t properties{};
  if (!check(gpuGetDevice(&device), "gpuGetDevice")) return 3;
  if (!check(gpuGetDeviceProperties(&properties, device), "gpuGetDeviceProperties")) {
    return 3;
  }
  int managed = 0;
  int concurrent_managed = 0;
  int pageable = 0;
  int host_page_tables = 0;
  if (!check(gpuDeviceGetAttribute(&managed, GPU_MANAGED_ATTRIBUTE, device),
             "managed_memory")) return 3;
  if (!check(gpuDeviceGetAttribute(
                 &concurrent_managed, GPU_CONCURRENT_MANAGED_ATTRIBUTE, device),
             "concurrent_managed_access")) return 3;
  if (!check(gpuDeviceGetAttribute(&pageable, GPU_PAGEABLE_ATTRIBUTE, device),
             "pageable_memory_access")) return 3;
  if (!check(gpuDeviceGetAttribute(
                 &host_page_tables, GPU_PAGE_TABLE_ATTRIBUTE, device),
             "pageable_memory_access_uses_host_page_tables")) return 3;

  const int threads = 256;
  const int blocks = properties.multiProcessorCount * 32;
  std::vector<CaseResult> cases;

  if (managed == 0) {
    cases.push_back(unavailable_case(
        "managed_unified", "cudaMallocManaged_or_hipMallocManaged",
        "coherent_managed_allocation", "managed_memory_attribute_false"));
  } else {
    Allocation allocation;
    const auto start = Clock::now();
    const bool allocated = allocate_managed(
        requested_bytes, concurrent_managed != 0, &allocation);
    const double seconds = elapsed_seconds(start);
    if (allocated) {
      cases.push_back(measure_case(
          "managed_unified", "cudaMallocManaged_or_hipMallocManaged",
          "coherent_managed_allocation", &allocation, seconds, blocks, threads,
          warmup, iterations, inner));
    } else {
      cases.push_back(unavailable_case(
          "managed_unified", "cudaMallocManaged_or_hipMallocManaged",
          "coherent_managed_allocation", "managed_allocation_failed"));
    }
  }

  if (pageable == 0 || host_page_tables == 0) {
    cases.push_back(unavailable_case(
        "system_pageable_direct", "posix_memalign",
        "system_pageable_direct_hmm_or_ats",
        pageable == 0 ? "pageable_memory_access_attribute_false"
                      : "host_page_tables_attribute_false"));
  } else {
    Allocation allocation;
    const auto start = Clock::now();
    const bool allocated = allocate_pageable(requested_bytes, &allocation);
    const double seconds = elapsed_seconds(start);
    if (allocated) {
      cases.push_back(measure_case(
          "system_pageable_direct", "posix_memalign",
          "system_pageable_direct_hmm_or_ats", &allocation, seconds, blocks,
          threads, warmup, iterations, inner));
    } else {
      cases.push_back(unavailable_case(
          "system_pageable_direct", "posix_memalign",
          "system_pageable_direct_hmm_or_ats", "pageable_allocation_failed"));
    }
  }

  Allocation vmm;
  const auto vmm_start = Clock::now();
  const bool vmm_allocated = allocate_vmm(device, requested_bytes, &vmm);
  const double vmm_seconds = elapsed_seconds(vmm_start);
  if (vmm_allocated) {
    cases.push_back(measure_case(
        "platform_vmm", "cudaMem_or_hipMem_vmm",
        "platform_virtual_memory_device_mapping", &vmm, vmm_seconds, blocks,
        threads, warmup, iterations, inner));
  } else {
    cases.push_back(unavailable_case(
        "platform_vmm", "cudaMem_or_hipMem_vmm",
        "platform_virtual_memory_device_mapping", "vmm_allocation_failed"));
  }

  std::cout << "{\"backend\":\"" << GPU_BACKEND
            << "\",\"device_name\":\"" << json_escape(properties.name)
            << "\",\"total_global_memory_bytes\":"
            << static_cast<unsigned long long>(properties.totalGlobalMem)
            << ",\"multiprocessor_count\":" << properties.multiProcessorCount
            << ",\"threads_per_block\":" << threads
            << ",\"blocks\":" << blocks
            << ",\"requested_buffer_bytes\":" << requested_bytes
            << ",\"capabilities\":{\"managed_memory\":" << managed
            << ",\"concurrent_managed_access\":" << concurrent_managed
            << ",\"pageable_memory_access\":" << pageable
            << ",\"pageable_memory_access_uses_host_page_tables\":"
            << host_page_tables << "},\"cases\":[";
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (index) std::cout << ',';
    print_case(cases[index]);
  }
  std::cout << "]}\n";
  return 0;
}
