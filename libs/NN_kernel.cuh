
#pragma once

#include <cuda_runtime.h>

// function to compute the inut of NN
void launch_vision(                 
    float* world_value,             
    int* world_id,                  
    float* world_signaling,        
    int dim_world,                 
    int* cell_idx,                 
    int dim_window,                       
    float* input_workspace,               
    int limit_workspace_cell,
    cudaStream_t stream
);

// function that compute the value between the layer of NN
void launch_NN_forward(                           
    float* input_workspace,                  
    float* output_workspace,                   
    int workspace_size,
    float* weights,                               
    int n_weights,                                
    float* biases,                                 
    int n_biases,                                   
    int* structure,   
    int limit_workspace_cell,
    int *cells,                                     
    int *world_id,                                 
    int dim_structure,                              
    cudaStream_t stream    
);

// function that elaborate the output of NN
void launch_output_elaboration(              
    float* world_value,                      
    float* world_signal,                     
    int* world_id,                        
    float* contribution_matrix,           
    float* output_workspace,              
    int* cells,                           
    int world_dim,                        
    int number_of_creatures,              
    int output_size,   
    int limit_workspace_cell,
    float energy_fraction,
    cudaStream_t stream
);

// function that compute the energy and occupation of the creature
void launch_compute_energy_and_occupation(
    float* world_value,
    int* world_id,
    float* occupation_vector,
    float* energy_vector,
    int world_dim,
    int n_creature,
    cudaStream_t stream
);

// function for recombine the models
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