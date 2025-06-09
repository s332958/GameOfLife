// utils_kernel.cuh
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Add random value in a vector, with value between [minVal, maxVal]
void launch_fill_random_kernel(float* d_vec, int start, int finish, float minVal, float maxVal,
                               curandState states[],cudaStream_t stream);

void launch_init_curandstates(curandState d_states[], int total_threads, unsigned long seed, cudaStream_t stream);