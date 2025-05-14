#include "Cellula.cuh"
#include <cuda_runtime.h>

// Kernel CUDA

__global__ void kernel_visione(float* mondo_cu, float* mondo_signal_cu, int dim_mondo){
    if (threadid.x > dim_visione || threadid.y > dim_visione) return;
    radius_filter = dim_finestraVisione / 2;

    int filtro_x = (threadIdx.x - radius_filter + centro_x) % dim_mondo;
    int filtro_y = (threadIdx.y - radius_filter + centro_y) % dim_mondo;
    int filtro_index = filtro_y * dim_world + filtro_x;

    int visione_index = filtro_y * dim_visione + filtro_x;
    visione[visione_index] = mondo_cu[filtro_index];
    visione[visione_index + dim_visione * dim_visione] = mondo_signal_cu[filtro_index];
}


// Funzione membro che lancia il kernel
void Cellula::launch_calcolo_visione(float* mondo_cu, float* mondo_signal_cu, int dim_mondo) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    //computation number of thread and block for launch kernel (use max thread for dimension before launch new block)
    int n_thread_per_block = properties.maxThreadsPerBlock;
    int thread_per_dimension = sqrt(n_thread_per_block);
    int n_block = dim_world / thread_per_dimension;
    if(n_block%thread_per_dimension!=0 || n_block==0) n_block=n_block+1

    dim3 thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    dim3 block_number = dim3(n_block, n_block); 
    kernel_visione<<<block_number,thread_number,0,stream>>>(mondo_cu, mondo_signal_cu, dim_mondo);
    cudaDeviceSynchronize();  // opzionale per il debug
}




