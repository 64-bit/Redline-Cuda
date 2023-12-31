#include "CudaUtils.h"

void Redline::CudaUtils::InitCuda()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  SM Count: %d\n",
            prop.multiProcessorCount);

        printf("\n");
    }

	cudaChecked(cudaSetDevice(0));
}

void Redline::CudaUtils::ShutdownCuda()
{
	cudaChecked(cudaDeviceReset());
}