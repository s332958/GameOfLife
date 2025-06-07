// utils_kernel.cuh
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Add random value in a vector, with value between [minVal, maxVal]
void launch_fill_random_kernel(float* d_vec, int start, int finish, float minVal, float maxVal,
                               unsigned long seed, cudaStream_t stream);



template <typename T>
void launch_reset_kernel(T* d_vec, int n, cudaStream_t stream = 0);




// NEW 


void launch_update_model(
    float *weight_starting_model,
    float *biases_starting_model,
    float *varation_weights_vector,
    float *varation_biases_vector,
    float *score_vector,
    int    n_weights,
    int    n_biases,
    int    n_creature,
    float  alpha,
    float  std,
    cudaStream_t stream
);

void launch_generate_clone_creature(
    float *weight_starting_model,
    float *biases_starting_model,
    float *weights_vector,
    float *biases_vector,
    float *varation_weights_vector,
    float *varation_biases_vector,
    int    n_weights,
    int    n_biases,
    int    n_creature,
    float  std,
    cudaStream_t stream,
    curandState_t *states
);

void launch_init_curandstates(curandState d_states[], int total_threads, unsigned long seed, cudaStream_t stream);