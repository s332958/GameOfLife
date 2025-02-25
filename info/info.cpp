#include <cuda_runtime.h>
#include <iostream>

void printCudaDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "Device " << device << ": " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes\n";
    std::cout << "Registers per Block: " << prop.regsPerBlock << "\n";
    std::cout << "Warp Size: " << prop.warpSize << "\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Number of Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "Clock Rate: " << prop.clockRate / 1000 << " MHz\n";
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz\n";
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
    std::cout << "Max Grid Size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << "\n";
    std::cout << "Max Threads Dim: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << "\n";
    std::cout << "Texture Alignment: " << prop.textureAlignment << " bytes\n";
    std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << "\n";
    std::cout << "ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << "\n";
    std::cout << "PCI Bus ID: " << prop.pciBusID << ", PCI Device ID: " << prop.pciDeviceID << "\n";
    std::cout << "PCI Domain ID: " << prop.pciDomainID << "\n";
    std::cout << "-------------------------------\n";
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        printCudaDeviceProperties(i);    // Leggo le statistiche della mia GPU
    cudaDeviceProp attributes;
    cudaGetDeviceProperties_v2(&attributes,0);
    }
    
    return 0;
}