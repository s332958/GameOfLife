
#pragma once

#include <cuda_runtime.h>

void launch_vision(
    float* world_value,
    int* world_id,
    float* world_signaling,
    int dim_world,
    int* cell_idx,
    int raggio,
    float* input_workspace_addr,
    cudaStream_t stream
);

void launch_NN_forward(
    float* input_workspace_addr,
    float* output_workspace_addr,
    float* weights,
    int n_weights,
    float* biases,
    int n_biases,
    int* structure,
    int cell_index,
    int* cells,
    int* world_id,
    int dim_structure,
    cudaStream_t stream
);

void launch_output_elaboration(
    float* world_value,
    float* world_signal,
    int* world_id,
    float* contribution_matrix,
    float* output_workspace_addr,
    int* cells,
    int world_dim,
    int number_of_creatures,
    int output_size,
    int cell_index,
    cudaStream_t stream
);

void launch_compute_energy_and_occupation(
    float* world_value,
    int* world_id,
    int* occupation_vector,
    float* energy_vector,
    int world_dim,
    int n_creature,
    cudaStream_t stream
);

void launch_recombine_models_kernel(
    float *d_weights, float *d_biases,
    float *d_new_weights, float *d_new_biases,
    int num_weights_per_model, int num_bias_per_model,
    int model1_idx, int model2_idx, int output_idx,
    float gen_x_block,
    float mutation_prob,
    float mutation_range,
    unsigned long seed,
    cudaStream_t stream
);