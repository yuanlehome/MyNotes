#include "common.h"
#include "kernel_caller_declare.h"

void deviceQuery() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    std::printf("There are no available device(s) that support CUDA\n");
  } else {
    std::printf("Detected %d CUDA Capable device(s)\n", device_count);
  }

  int device_id = 0;
  CUDA_CHECK(cudaSetDevice(device_id));
  cudaDeviceProp device_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));

  std::printf("\nDevice %d: \"%s\"\n", device_id, device_prop.name);

  int driver_version = 0;
  int runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  std::printf(
      "  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
      driver_version / 1000,
      (driver_version % 100) / 10,
      runtime_version / 1000,
      (runtime_version % 100) / 10);
  std::printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
              device_prop.major,
              device_prop.minor);

  std::printf(
      "  Total amount of global memory:                 %lu MBytes (%lu "
      "bytes)\n",
      device_prop.totalGlobalMem / 1024 / 1024,
      device_prop.totalGlobalMem);

  std::printf(
      "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
      "GHz)\n",
      device_prop.clockRate * 1e-3f,
      device_prop.clockRate * 1e-6f);

  std::printf("  Memory Clock rate:                             %.0f Mhz\n",
              device_prop.memoryClockRate * 1e-3f);
  std::printf("  Memory Bus Width:                              %d-bit\n",
              device_prop.memoryBusWidth);

  if (device_prop.l2CacheSize) {
    std::printf("  L2 Cache Size:                                 %d bytes\n",
                device_prop.l2CacheSize);
  }

  std::printf("  Total amount of constant memory:               %zu bytes\n",
              device_prop.totalConstMem);
  std::printf("  Total amount of shared memory per block:       %zu bytes\n",
              device_prop.sharedMemPerBlock);
  std::printf("  Total shared memory per multiprocessor:        %zu bytes\n",
              device_prop.sharedMemPerMultiprocessor);
  std::printf("  Total number of registers available per block: %d\n",
              device_prop.regsPerBlock);
  std::printf("  Warp size:                                     %d\n",
              device_prop.warpSize);
  std::printf("  Maximum number of threads per multiprocessor:  %d\n",
              device_prop.maxThreadsPerMultiProcessor);
  std::printf("  Maximum number of threads per block:           %d\n",
              device_prop.maxThreadsPerBlock);
  std::printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
              device_prop.maxThreadsDim[0],
              device_prop.maxThreadsDim[1],
              device_prop.maxThreadsDim[2]);
  std::printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
              device_prop.maxGridSize[0],
              device_prop.maxGridSize[1],
              device_prop.maxGridSize[2]);
  std::printf("  Maximum memory pitch:                          %zu bytes\n",
              device_prop.memPitch);
  std::printf("  Texture alignment:                             %zu bytes\n",
              device_prop.textureAlignment);
  std::printf(
      "  Concurrent copy and kernel execution:          %s with %d copy "
      "engine(s)\n",
      (device_prop.deviceOverlap ? "Yes" : "No"),
      device_prop.asyncEngineCount);
  std::printf("  Run time limit on kernels:                     %s\n",
              device_prop.kernelExecTimeoutEnabled ? "Yes" : "No");
  std::printf("  Integrated GPU sharing Host Memory:            %s\n",
              device_prop.integrated ? "Yes" : "No");
  std::printf("  Support host page-locked memory mapping:       %s\n",
              device_prop.canMapHostMemory ? "Yes" : "No");
  std::printf("  Alignment requirement for Surfaces:            %s\n",
              device_prop.surfaceAlignment ? "Yes" : "No");
  std::printf("  Device has ECC support:                        %s\n",
              device_prop.ECCEnabled ? "Enabled" : "Disabled");
  std::printf("  Device supports Unified Addressing (UVA):      %s\n",
              device_prop.unifiedAddressing ? "Yes" : "No");
  std::printf("  Device supports Managed Memory:                %s\n",
              device_prop.managedMemory ? "Yes" : "No");
  std::printf("  Device supports Compute Preemption:            %s\n",
              device_prop.computePreemptionSupported ? "Yes" : "No");
  std::printf("  Supports Cooperative Kernel Launch:            %s\n",
              device_prop.cooperativeLaunch ? "Yes" : "No");
  std::printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
              device_prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
  std::printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
              device_prop.pciDomainID,
              device_prop.pciBusID,
              device_prop.pciDeviceID);

  cudaSharedMemConfig config;
  cudaDeviceGetSharedMemConfig(&config);
  const char *sm_bank_size[] = {"Default", "FourByte", "EightByte"};
  std::printf("  The current size of shared memory banks:       %s\n",
              sm_bank_size[static_cast<int>(config)]);

  const char *compute_mode[] = {
      "Default (multiple host threads can use ::cudaSetDevice() with device "
      "simultaneously)",
      "Exclusive (only one host thread in one process is able to use "
      "::cudaSetDevice() with this device)",
      "Prohibited (no host thread can use ::cudaSetDevice() with this "
      "device)",
      "Exclusive Process (many threads in one process is able to use "
      "::cudaSetDevice() with this device)",
      "Unknown",
      NULL};
  std::printf("  Compute Mode:\n");
  std::printf("     < %s >\n", compute_mode[device_prop.computeMode]);
}
