#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp c;
    int device = 0;

    // Ottieni le proprietà del dispositivo 0 (il primo dispositivo GPU)
    cudaError_t err = cudaGetDeviceProperties(&c, device);
    if (err != cudaSuccess) {
        printf("Errore nel recupero delle proprietà del dispositivo: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Stampa alcune proprietà principali della GPU
    printf("Nome del dispositivo: %s\n", c.name);
    printf("Numero di multiprocessori: %d\n", c.multiProcessorCount);
    printf("Memoria globale disponibile: %lu bytes\n", c.totalGlobalMem);
    printf("Memoria condivisa per blocco: %lu bytes\n", c.sharedMemPerBlock);  // Memoria condivisa
    printf("Massima dimensione della griglia: (%d, %d, %d)\n", 
           c.maxGridSize[0], c.maxGridSize[1], c.maxGridSize[2]);
    printf("Massima dimensione del blocco: (%d, %d, %d)\n", 
           c.maxThreadsDim[0], c.maxThreadsDim[1], c.maxThreadsDim[2]);
    printf("Numero massimo di thread per blocco: %d\n", c.maxThreadsPerBlock);
    printf("Compatibilità del calcolo (major.minor): %d.%d\n", c.major, c.minor);

    return 0;
}
