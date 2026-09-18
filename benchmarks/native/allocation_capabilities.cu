#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

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
#else
#include <cuda_runtime.h>
#define GPU_BACKEND "cuda"
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#define gpuDeviceProp_t cudaDeviceProp
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

template <typename Attribute>
bool emit_attribute(
    const char* name, Attribute attribute, int device, bool* first) {
  int value = 0;
  if (!check(gpuDeviceGetAttribute(&value, attribute, device), name)) return false;
  if (!*first) std::cout << ',';
  *first = false;
  std::cout << '"' << name << "\":" << value;
  return true;
}

}  // namespace

int main() {
  int device = 0;
  gpuDeviceProp_t properties{};
  if (!check(gpuGetDevice(&device), "gpuGetDevice")) return 2;
  if (!check(gpuGetDeviceProperties(&properties, device), "gpuGetDeviceProperties")) {
    return 2;
  }

  std::cout << "{\"backend\":\"" << GPU_BACKEND << "\",\"device_name\":\""
            << json_escape(properties.name) << "\",\"total_global_memory_bytes\":"
            << static_cast<unsigned long long>(properties.totalGlobalMem)
            << ",\"attributes\":{";
  bool first = true;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (!emit_attribute("managed_memory", hipDeviceAttributeManagedMemory, device, &first)) return 3;
  if (!emit_attribute("concurrent_managed_access", hipDeviceAttributeConcurrentManagedAccess, device, &first)) return 3;
  if (!emit_attribute("pageable_memory_access", hipDeviceAttributePageableMemoryAccess, device, &first)) return 3;
  if (!emit_attribute("pageable_memory_access_uses_host_page_tables", hipDeviceAttributePageableMemoryAccessUsesHostPageTables, device, &first)) return 3;
  if (!emit_attribute("direct_managed_memory_access_from_host", hipDeviceAttributeDirectManagedMemAccessFromHost, device, &first)) return 3;
  if (!emit_attribute("host_native_atomic_supported", hipDeviceAttributeHostNativeAtomicSupported, device, &first)) return 3;
  if (!emit_attribute("memory_pools_supported", hipDeviceAttributeMemoryPoolsSupported, device, &first)) return 3;
  if (!emit_attribute("virtual_memory_management_supported", hipDeviceAttributeVirtualMemoryManagementSupported, device, &first)) return 3;
#else
  if (!emit_attribute("managed_memory", cudaDevAttrManagedMemory, device, &first)) return 3;
  if (!emit_attribute("concurrent_managed_access", cudaDevAttrConcurrentManagedAccess, device, &first)) return 3;
  if (!emit_attribute("pageable_memory_access", cudaDevAttrPageableMemoryAccess, device, &first)) return 3;
  if (!emit_attribute("pageable_memory_access_uses_host_page_tables", cudaDevAttrPageableMemoryAccessUsesHostPageTables, device, &first)) return 3;
  if (!emit_attribute("direct_managed_memory_access_from_host", cudaDevAttrDirectManagedMemAccessFromHost, device, &first)) return 3;
  if (!emit_attribute("host_native_atomic_supported", cudaDevAttrHostNativeAtomicSupported, device, &first)) return 3;
  if (!emit_attribute("memory_pools_supported", cudaDevAttrMemoryPoolsSupported, device, &first)) return 3;
#endif
  std::cout << "}}\n";
  return 0;
}
