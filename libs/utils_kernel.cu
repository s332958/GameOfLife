#include "utils_kernel.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <iostream>



// ============================================================================

__global__ void fill_random_kernel(float* d_vec, int start, int finish, float minVal, float maxVal, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= finish - start) return;

    int real_idx = idx + start;

    // Setup generator per thread
    curandState state;
    curand_init(seed, idx, 0, &state);

    float rand_uniform = curand_uniform(&state); // [0,1)

    d_vec[real_idx] = minVal + rand_uniform * (maxVal - minVal);
}



// Wrapper: fill vettore random con calcolo griglia/thread
void launch_fill_random_kernel(float* d_vec, int start, int finish, float minVal, float maxVal,
                                unsigned long seed, cudaStream_t stream) {

    int threads = 1024;
    int n = finish - start;
    int blocks = (n + threads - 1) / threads;

    fill_random_kernel<<<blocks, threads, 0, stream>>>(d_vec, start, finish, minVal, maxVal, seed);

}
template <typename T>
__global__ void resetKernel(T* d_vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_vec[idx] = T(0);
}

template <typename T>
void launch_reset_kernel(T* d_vec, int n, cudaStream_t stream) {
    int threads = 1024;
    int blocks = (n + threads - 1) / threads;
    resetKernel<T><<<blocks, threads, 0, stream>>>(d_vec, n);
}

// ========== Esplicit template instantiation ==========
template void launch_reset_kernel<float>(float*, int, cudaStream_t);
template void launch_reset_kernel<int>(int*, int, cudaStream_t);
