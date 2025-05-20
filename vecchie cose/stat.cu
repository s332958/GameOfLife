#include <cuda_runtime.h>
#include <stdio.h>

void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    
    if (err != cudaSuccess) {
        printf("Errore nel recupero delle proprietà del dispositivo: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("\n============================================\n");
    printf("  📌 Informazioni sulla GPU (Device %d)\n", device);
    printf("============================================\n");

    printf("🔹 Nome del dispositivo: %s\n", prop.name);
    printf("🔹 Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("🔹 Multiprocessori (SMs): %d\n", prop.multiProcessorCount);
    printf("🔹 Frequenza di clock: %.2f MHz\n", prop.clockRate / 1000.0);
    
    printf("\n💾 Memoria:\n");
    printf("-------------------------------\n");
    printf("🔸 Memoria globale: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("🔸 Memoria condivisa per blocco: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("🔸 Memoria costante: %.2f KB\n", prop.totalConstMem / 1024.0);
    printf("🔸 Larghezza della banda della memoria: %d-bit\n", prop.memoryBusWidth);

    printf("\n🧵 Thread & Blocchi:\n");
    printf("-------------------------------\n");
    printf("🔸 Numero massimo di thread per blocco: %d\n", prop.maxThreadsPerBlock);
    printf("🔸 Dimensione massima del blocco: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("🔸 Dimensione massima della griglia: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    printf("\n📌 Proprietà avanzate:\n");
    printf("-------------------------------\n");
    printf("🔸 Warp size: %d\n", prop.warpSize);
    printf("🔸 Numero massimo di warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("🔸 Concurrent Kernels: %s\n", prop.concurrentKernels ? "Sì" : "No");
    printf("🔸 Unified Addressing: %s\n", prop.unifiedAddressing ? "Sì" : "No");

    printf("\n============================================\n");
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("Nessuna GPU CUDA disponibile!\n");
        return -1;
    }

    for (int i = 0; i < deviceCount; i++) {
        printDeviceProperties(i);
    }

    return 0;
}
