#include "NN_kernel.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

// some activation function
__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ float fast_sigmoid(float x) {
    return 0.5f * (x / (1.0f + fabsf(x))) + 0.5f;  
}

// ============================================================================

__global__ void vision_kernel(
    float* world_value,
    int* world_id,
    float* world_signaling,
    int dim_world,
    int* cell_idx,
    int dim_window,
    float* workspace,
    int limit_workspace_cell,
    int dim_input
) 
{
    //global index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    //index of workspace
    int index_cell = index / dim_input;
    //cell of the workspace
    int vision_index = index % dim_input;
    
    //return thread that exceed 
    if (index >= dim_input*limit_workspace_cell) return;

    // dimension of vision of each alive cell
    int dim_window_sq = dim_window*dim_window;
    int radius = dim_window / 2;

    //get the center
    int center_index = cell_idx[index_cell];

    // indixing all window vision positions
    int vision_window_index = vision_index % dim_window_sq;
    int visiontype_index = vision_index / dim_window_sq;

    int center_x = center_index % dim_world;
    int center_y = center_index / dim_world;

    int vision_x = vision_window_index % dim_window;
    int vision_y = vision_window_index / dim_window;

    int real_vision_x = (center_x + (vision_x - radius) + dim_world) % dim_world;
    int real_vision_y = (center_y + (vision_y - radius) + dim_world) % dim_world;

    int real_vision_index = real_vision_y*dim_world + real_vision_x;

    int cell_ID = world_id[center_index];
    int vision_ID = world_id[real_vision_index];

    // apply the change of sign for enemy cell
    if(visiontype_index == 0){
        if(cell_ID == vision_ID){            
            workspace[index] = world_signaling[real_vision_index];
        }else{
            workspace[index] = - world_signaling[real_vision_index];
        }

    }else{
        if(vision_ID == -1){
            workspace[index] = - 1.0f;
        }else{
            workspace[index] = world_value[real_vision_index];
        }

    }
}

// ============================================================================


__global__ void NN_forward_weight_kernel(
    float* input,
    float* output, 
    float* weights, 
    int* cells,
    int* world_id, 
    int n_weights, 
    int limit_workspace_cell,
    int layer1_size,
    int layer2_size,
    int offset_weights
){
        // index del thread 
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;

        // se il thread supera la dim dei weights  allora ritorno 
        if (tidx >= layer1_size * layer2_size * limit_workspace_cell){
            return;
        }
        // cell index is the index of the cell in the array alive cell
        int cell_index = tidx / (layer1_size * layer2_size);
        // index of the weight 
        int weight_index = tidx % (layer1_size * layer2_size);
        
        // get the id from cell index
        int ID = world_id[cells[cell_index]];
        
        // get the correct weight of the correct creature, consisdering the layer offset
        int true_weight_index = n_weights * (ID - 1) + weight_index + offset_weights; 
        
        // indexing for compute the input and output
        int input_neuron_idx  = weight_index % layer1_size;
        int output_neuron_idx = weight_index / layer1_size;

        // consider position of cell in the input or output array and then get the offset of the correct input or output
        int input_index  = cell_index * layer1_size  + input_neuron_idx;
        int output_index = cell_index * layer2_size  + output_neuron_idx;
        
        // compute the result value
        float weighted = weights[true_weight_index] * input[input_index];

        // atomic add for update the corrispondent output cell in the corrispondent working station
        atomicAdd(&output[output_index], weighted);
        
}

__global__ void NN_forward_bias_kernel(
    float* output, 
    float* biases, 
    int* cells,
    int* world_id, 
    int n_biases, 
    int limit_workspace_cell,
    int layer2_size,
    int offset_biases
){
        // index del thread 
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;
       
        if (tidx >= layer2_size * limit_workspace_cell){
            return;
        }
        int cell_index = tidx / layer2_size;
        int bias_index = tidx % layer2_size;

        int world_index = cells[cell_index];
        int ID = world_id[world_index];

        int true_bias_index = n_biases * (ID - 1) + bias_index + offset_biases;

        output[tidx] += biases[true_bias_index];

        // applico la relu ad ogni cella di output modificata 
        output[tidx] = fast_sigmoid(output[tidx]);
        
}

//===================================================================================


__global__ void output_elaboration_kernel(
    float* world_value,
    float* world_signal,
    int* world_id,
    float* contribution_matrix,
    float* outputs,   
    int* cells,
    int dim_world, 
    int number_of_creatures,
    int output_size,
    int dim_window,
    int limit_workspace_cell,
    float energy_fraction
){  
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index_cell = index / output_size;
    int output_index = index % output_size;
    
    if (index >= output_size*limit_workspace_cell) return;
    
    int center_index = cells[index_cell];

    float output = outputs[index];
    output = sigmoid(output);

    if (output_index == (output_size - 1)){
        world_signal[center_index] = output;
        return;
    }

    float center_value = world_value[center_index];

    int dim_window_sq = dim_window*dim_window;
    float final_output = center_value * energy_fraction / (dim_window_sq - 1) * output;
    int radius = dim_window / 2;

    int vision_window_index = output_index % dim_window_sq;

    int center_x = center_index % dim_world;
    int center_y = center_index / dim_world;

    int vision_x = vision_window_index % dim_window;
    int vision_y = vision_window_index / dim_window;

    int real_vision_x = (center_x + (vision_x - radius) + dim_world) % dim_world;
    int real_vision_y = (center_y + (vision_y - radius) + dim_world) % dim_world;

    int cell_ID = world_id[center_index];
    int ID_offset = (cell_ID - 1) * (dim_world * dim_world);

    int real_vision_index = real_vision_y*dim_world + real_vision_x + ID_offset;

    atomicAdd(&contribution_matrix[real_vision_index], final_output);
    atomicAdd(&world_value[center_index], - final_output); 
}

// ===================================================================================

__global__ void compute_energy_and_occupation_kernel(
    float* world_value,
    int* world_id,
    float* occupation_vector,
    float* energy_vector,
    int world_dim,
    int n_creature
) {
    //shared memmory with ncreature*2
    extern __shared__ float shared_mem[]; 
    float* shared_occ = shared_mem;
    float* shared_energy = shared_mem + n_creature;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = world_dim * world_dim;

    // Initialize shared memory
    if (tid < n_creature) {
        shared_occ[tid] = 0.0f;
        shared_energy[tid] = 0.0f;
    }
    __syncthreads();

    // Each thread processes one cell
    if (idx < total_cells) {
        int id = world_id[idx] - 1;
        if (id >= 0 && id < n_creature) {
            atomicAdd(&shared_occ[id], 1.0f);
            atomicAdd(&shared_energy[id], world_value[idx]);
        }
    }

    __syncthreads();

    // One warp (or all threads) writes back to global memory
    if (tid < n_creature) {
        atomicAdd(&occupation_vector[tid], shared_occ[tid]);
        atomicAdd(&energy_vector[tid], shared_energy[tid]);
    }
}


//=================================================================================

// Wrapper kernel visione
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
){

    int n_thread_per_block = 1024;
    int dim_input = dim_window * dim_window * 2;
    int thread_number = dim_input * limit_workspace_cell;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;
    vision_kernel<<<n_block,n_thread_per_block,0,stream>>>(
        world_value,
        world_id,
        world_signaling,
        dim_world,
        cell_idx,
        dim_window,
        input_workspace,
        limit_workspace_cell,
        dim_input
    ); 

}

// ===================================================================================================

// Wrapper kernel NN_forward
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
){
    int n_thread_per_block = 1024;
    int layer1_size = 0;
    int layer2_size = 0;
    int weight_offset = 0;
    int biases_offset = 0;

    for(int i=0; i < (dim_structure-1); i++){

        layer1_size = structure[i];
        layer2_size = structure[i + 1];

        int thread_number = layer1_size * layer2_size * limit_workspace_cell;             

        int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;

        NN_forward_weight_kernel<<<n_block, n_thread_per_block, 0 , stream>>>(
            input_workspace,
            output_workspace, 
            weights, 
            cells, 
            world_id, 
            n_weights, 
            limit_workspace_cell,
            layer1_size, 
            layer2_size, 
            weight_offset
        );  

        thread_number = layer2_size * limit_workspace_cell;             
        n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;


        NN_forward_bias_kernel<<<n_block, n_thread_per_block, 0 , stream>>>(
            output_workspace, 
            biases, 
            cells, 
            world_id, 
            n_biases, 
            limit_workspace_cell,
            layer2_size, 
            biases_offset
        ); 
        
        if (i < (dim_structure-2)){
            // reset all workspaces
            cudaMemcpy(input_workspace, output_workspace, workspace_size*limit_workspace_cell, cudaMemcpyDeviceToDevice);
            cudaMemset(output_workspace, 0, workspace_size*limit_workspace_cell);
        }
        weight_offset += layer1_size*layer2_size;
        biases_offset += layer2_size;

    }

}

// ===================================================================================================

//Wrapper kernel output_elaboration
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
){
    int window_size = sqrt(output_size - 1);
    int n_thread_per_block = 1024;
    int thread_number = limit_workspace_cell * output_size;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;

    output_elaboration_kernel<<<n_block, n_thread_per_block, 0, stream>>>(
        world_value,
        world_signal,
        world_id,
        contribution_matrix,
        output_workspace,
        cells,
        world_dim,
        number_of_creatures,
        output_size,
        window_size,
        limit_workspace_cell,
        energy_fraction
    );
    

}

// ===================================================================================================

// Wrapper compute energy and occupation for evaluation
void launch_compute_energy_and_occupation(
    float* world_value,
    int* world_id,
    float* occupation_vector,
    float* energy_vector,
    int world_dim,
    int n_creature,
    cudaStream_t stream
){

    int n_thread_per_block = 1024; 
    int thread_number = world_dim * world_dim;

    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;
    size_t shared_memory_size = n_creature*2*sizeof(float);

    compute_energy_and_occupation_kernel<<<n_block,n_thread_per_block,shared_memory_size,stream>>>(
        world_value,
        world_id,
        occupation_vector,
        energy_vector,
        world_dim,
        n_creature
    );


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
    int    limit_creature,
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

    // the limit creature impose how many creature depends from original, model, the rest are regenate random.
    if(id_creature<limit_creature){
        varation_weights_vector[final_pos] = varation;
        weights_vector[final_pos] = weight_starting_model[param_original_idx] + varation;
    }else{
        varation_weights_vector[final_pos] = varation - weight_starting_model[param_original_idx];
        weights_vector[final_pos] = varation;
    }


    if(idx >= n_creature*n_biases) return;

    varation = (curand_uniform(&state) * 2) -1; 
    varation = varation * std; 

    id_creature = idx / n_biases;
    param_original_idx = idx % n_biases;
    final_pos = id_creature*n_biases + param_original_idx;

    // the limit creature impose how many creature depends from original, model, the rest are regenate random.
    if(id_creature<limit_creature){
        varation_biases_vector[final_pos] = varation;
        biases_vector[final_pos] = biases_starting_model[param_original_idx] + varation;
    }else{
        varation_biases_vector[final_pos] = varation - biases_starting_model[param_original_idx];
        biases_vector[final_pos] = varation;
    }


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
    int    limit_creature,
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
        limit_creature,
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
    float  std,
    int n_steps
){

    // 0 -> take the contribution of each creature
    // 1 -> take total points of each creature for normalizations
    __shared__ float shared_mem[2];

    int creature_idx = threadIdx.x;
    int params_idx = blockIdx.x;

    if(creature_idx >= n_creature || params_idx >= n_biases+n_weights) return;

    if(threadIdx.x==0){
        shared_mem[0] = 0;
        shared_mem[1] = 0;
    }

    __syncthreads();

    if(blockIdx.x < n_weights){

        // cumulate points for each creature
        atomicAdd(&shared_mem[1],score_vector[creature_idx]);
        __syncthreads();

        // cumulate contribution for each creature
        float val = varation_weights_vector[params_idx] * score_vector[creature_idx] / shared_mem[1];
        atomicAdd(&shared_mem[0],val);

        __syncthreads();

        val = shared_mem[0];

        val = (val * alpha) / (n_creature * std);
        weight_starting_model[params_idx] += val;

    }else{

        params_idx -= n_weights;

        atomicAdd(&shared_mem[1],score_vector[creature_idx]);
        __syncthreads();

        float val = varation_biases_vector[params_idx] * score_vector[creature_idx] / shared_mem[1];
        atomicAdd(&shared_mem[0],val);

        __syncthreads();

        val = shared_mem[0];

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
    int n_steps,
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
        std,
        n_steps
    );

}