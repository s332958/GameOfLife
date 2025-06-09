
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

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

// Function for update the original model
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
    int n_steps,
    cudaStream_t stream
);

// function for generating creature from the first
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
    int    limit_creature,
    float  std,
    cudaStream_t stream,
    curandState_t *states
);