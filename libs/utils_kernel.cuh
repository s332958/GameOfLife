// utils_kernel.cuh
#pragma once

#include <cuda_runtime.h>

// Riempie un vettore di float con valori casuali in un intervallo [minVal, maxVal]
void launch_fill_random_kernel(float* d_vec, int n, float minVal, float maxVal,
                               unsigned long seed, cudaStream_t stream);



template <typename T>
void launch_reset_kernel(T* d_vec, int n, cudaStream_t stream = 0);