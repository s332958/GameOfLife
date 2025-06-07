#include "utils_kernel.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <iostream>



// ============================================================================

// funtion for generating random number in a vector in GPU
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































































// NUOVI TEST





__global__ void init_curandstates_kernel(curandState* states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

void launch_init_curandstates(curandState d_states[], int total_threads, unsigned long seed, cudaStream_t stream) {
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    init_curandstates_kernel<<<blocks, threads_per_block, 0, stream>>>(d_states, seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in init_curandstates_kernel: %s\n", cudaGetErrorString(err));
    }
}

// ================================================================================================

__global__ void generate_clone_creature_kernel(
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
    curandState_t *states
){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= n_creature*n_weights) return;

    curandState state = states[threadIdx.x];

    float varation = (curand_uniform(&state) * 2) -1;
    varation = varation * std; 

    int id_creature = idx / n_weights;
    int param_original_idx = idx % n_weights;
    int final_pos = id_creature*n_weights + param_original_idx;

    varation_weights_vector[final_pos] = varation;
    weights_vector[final_pos] = weight_starting_model[param_original_idx] + varation;


    if(idx >= n_creature*n_biases) return;

    varation = (curand_uniform(&state) * 2) -1; 

    id_creature = idx / n_biases;
    param_original_idx = idx % n_biases;
    final_pos = id_creature*n_biases + param_original_idx;

    varation_biases_vector[final_pos] = varation;
    biases_vector[final_pos] = biases_starting_model[param_original_idx] + varation;


}


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
){

    int n_thread = n_weights*n_creature;
    if(n_thread>1024) n_thread = 1024;
    int n_block = (n_weights + n_thread -1) / n_thread;

    generate_clone_creature_kernel<<<n_block,n_thread,0,stream>>>(
        weight_starting_model,
        biases_starting_model,
        weights_vector,
        biases_vector,
        varation_weights_vector,
        varation_biases_vector,
        n_weights,
        n_biases,
        n_creature,
        std,
        states
    );

}



// ========================================================================================================


__global__ void update_model_kernel(
    float *weight_starting_model,
    float *biases_starting_model,
    float *varation_weights_vector,
    float *varation_biases_vector,
    float *score_vector,
    int    n_weights,
    int    n_biases,
    int    n_creature,
    float  alpha,
    float  std
){

    __shared__ float shared_mem;

    int creature_idx = threadIdx.x;
    int params_idx = blockIdx.x;

    if(creature_idx >= n_creature || params_idx >= n_biases+n_weights) return;

    if(threadIdx.x==0){
        shared_mem = 0;
    }

    __syncthreads();

    if(blockIdx.x < n_weights){

        int val = varation_weights_vector[params_idx] * score_vector[creature_idx];
        atomicAdd(&shared_mem,val);

        __syncthreads();

        val = shared_mem;

        val = (val * alpha) / (n_creature * std);
        weight_starting_model[params_idx] += val;

    }else{

        params_idx -= n_weights;

        int val = varation_biases_vector[params_idx] * score_vector[creature_idx];
        atomicAdd(&shared_mem,val);

        __syncthreads();

        val = shared_mem;

        val = (val * alpha) / (n_creature * std);
        biases_starting_model[params_idx] += val;

    }

}


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
){

    int n_thread = n_creature;
    if(n_thread>1024) n_thread = 1024;
    int n_block = n_weights+n_biases;

    update_model_kernel<<<n_block,n_thread,0,stream>>>(
        weight_starting_model,
        biases_starting_model,
        varation_weights_vector,
        varation_biases_vector,
        score_vector,
        n_weights,
        n_biases,
        n_creature,
        alpha,
        std
    );

}