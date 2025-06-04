#include "NN_kernel.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

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

// function that compute of surrounding of the alive cell, this is the input of the NN
__global__ void vision_kernel(
    float* world_value,
    int* world_id,
    float* world_signaling,
    int dim_world,
    int* cell_idx,
    int raggio,
    float* workspace_addr
) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y; 

    if (x >= raggio || y >= raggio) return;

    int cell_index = *cell_idx;
    int ID_cell = world_id[cell_index];

    int center_row = cell_index / dim_world;
    int center_col = cell_index % dim_world;

    // coumpute world sourrounding cells
    int world_row = (center_row - (raggio / 2) + y + dim_world) % dim_world;
    int world_col = (center_col - (raggio / 2) + x + dim_world) % dim_world;

    int offset = (y*raggio + x)*2;

    int world_pos = world_row * dim_world + world_col;
    int ID_vision = world_id[world_pos];

    workspace_addr[offset + 0] = world_value[world_pos];
    if(ID_cell == ID_vision){
        workspace_addr[offset + 1] = world_signaling[world_pos];
    }else{
        workspace_addr[offset + 1] = - world_signaling[world_pos];
    }
}

// ============================================================================

// function that compute the elaboration to one layer of NN to another
// ONE BLOCK LIMITATION
__global__ void NN_forward_kernel(
    float* input_addr,
    float* output_addr, 
    float* weights, 
    float* biases, 
    int cell_index, 
    int* cells,
    int* world_id, 
    int n_weights, 
    int n_biases, 
    int layer1_size, 
    int layer2_size,
    int offset_weights,
    int offset_biases
){
        //thread index
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;

        // if thread exceed dim weights return
        if (tidx >= layer2_size * layer1_size){
            return;
        }

        //obtain the ID of cell from world id, to select the model for computing output
        int world_index = cells[cell_index];
        int ID = world_id[world_index];

        //if (ID <= 0)return;

        // get index of the weights of the model
        int weight_index = n_weights * (ID - 1) + tidx + offset_weights; // + offset weights of the layer (es layer0 offeset=0, layer1 offset=layer0*layer1)
        
        //assing at each thread the cell where read the input and write the output
        int output_index = tidx % layer2_size;
        int input_index = tidx / layer2_size;
        
        // compute for each thread the weighted value
        float weighted = weights[weight_index] * input_addr[input_index];

        // clean the output cell 
        if (tidx < layer2_size){
            output_addr[tidx] = 0;
        }

        __syncthreads();

        // add for eanch output cell the weighted value
        atomicAdd(&output_addr[output_index], weighted);
        
        // block all the thread that have dimension higher than number of biases 
        if (tidx >= layer2_size){
            return;
        }
        
        // obtain the the biases index of the model
        int bias_index = n_biases * (ID - 1) + tidx + offset_biases;  // + offset becouse biases depend on layer (es layer1 offeset=0, layer2 offset=layer1)

        __syncthreads();

        // sum the bias to corret output cell
        output_addr[tidx] += biases[bias_index];
    
        // apply relu function to the output cell
        output_addr[tidx] = relu(output_addr[tidx]);

}

//===================================================================================

// this function compute the output obtain from the model layer
__global__ void output_elaboration_kernel(
    float* world_value,
    float* world_signal,
    int* world_id,
    float* contribution_matrix,
    float* outputs,   
    int* cells,
    int world_dim, 
    int number_of_creatures,
    int output_size,
    int cell_index,
    float frazione_di_se_stesso
){  
    // index thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // return if index thread is greater than output size
    if(index >= output_size) return;

    // get value from the output cell
    float output = outputs[index];

    // apply sigmoid
    output = sigmoid(output);

    //get information about alive cell (ID and value) and its position
    int center_index = cells[cell_index];
    int output_index = index;
    int ID = world_id[center_index];

    // last thread update world siganling
    if(output_index == output_size-1){
        world_signal[center_index] = output;
        return;
    }

    // get the value of the cell
    float center_value = world_value[center_index];

    // compute x ad y of the world (of the cell alive)
    int center_x = center_index % world_dim;
    int center_y = center_index / world_dim;

    // get the ray surrounding where write the output
    int dim_out_window = sqrtf(output_size-1);
    int raggio = dim_out_window/2;

    // get x y of cell surrounding considering the world toroidal
    int filter_x = ((output_index % dim_out_window) - raggio + center_x + world_dim)%world_dim;
    int filter_y = ((output_index / dim_out_window) - raggio + center_y + world_dim)%world_dim;
    
    // find index in contribution matrix for indicate the contribution of the creature in each cell
    int filter_index = (world_dim * world_dim * (ID - 1)) + (filter_y * world_dim) + filter_x;


    // compute final result in the cell of contribution
    float final_output = center_value * frazione_di_se_stesso * output;

    // apply contribution to the contribution matrix
    atomicAdd(&contribution_matrix[filter_index], final_output);

    // modifiy the value of the cell that generate the movment
    atomicAdd(&world_value[center_index], - final_output);    
}

// ===================================================================================

// function to compute the energy and occupation of each creature
__global__ void compute_energy_and_occupation_kernel(
    float* world_value,
    int* world_id,
    float* occupation_vector,
    float* energy_vector,
    int world_dim,
    int n_creature) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= world_dim * world_dim) return;

    int id = world_id[index] - 1;

    if (id < 0)return;
    
    atomicAdd(&occupation_vector[id], 1.0f/float(world_dim));
    atomicAdd(&energy_vector[id], world_value[index]);

}

// ==================================================================================

// Function that recombine the creature parameters
// NOTE:
// the block dim is important, the number of thread of eache block represent the number of genes in each genetic block
__global__ void recombine_models_kernel(
    float *weights, float *biases,
    float *new_weights, float *new_biases,
    int num_weights_per_model, int num_bias_per_model,
    int model1_idx, int model2_idx, int output_idx,
    float mutation_prob,
    float mutation_range,
    unsigned long seed)
{
    __shared__ int gen_id;

    //caompute thread and max number of genes in the model
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_genes = num_weights_per_model + num_bias_per_model;
    // return if the thread are greater of the sum
    if (idx >= total_genes) return;

    // creation of curand state for random value
    curandState state;
    curand_init(seed, idx, 0, &state);

    // the first thread define the parent of the genetic block
    if(threadIdx.x==0){
        int gen = curand(&state) % 2;
        gen_id = gen==0?model1_idx:model2_idx;
    }

    __syncthreads();

    int idx_model_param_gen = -1;

    // phase of weights computation (idx < num of weights)
    if (idx < num_weights_per_model) {

        // get index of gene form the parents parametes
        idx_model_param_gen = (gen_id * num_weights_per_model) + idx;

        // load gene value
        float gene_value = weights[idx_model_param_gen];

        // define a random value, if value is less than mutation probability, so the gene variate his value
        if (curand_uniform(&state) > mutation_prob) {
            //compute mutation value, is in range from -mutation_range to mutation_range
            float delta = (curand_uniform(&state) * 2.0f - 1.0f) * mutation_range;
            // apply mutation value
            gene_value += delta;
        }

        // find the index of the new gene and then update
        int idx_model_param_out = (output_idx * num_weights_per_model + idx);
        new_weights[idx_model_param_out] = gene_value;

    }else{

        //case biases (idx > weights model) 
        //compute index of bias gene considerig the offset of weights (biases threads are after the list of weights threads)
        idx_model_param_gen = (gen_id * num_bias_per_model) + idx - num_weights_per_model;

        // this part is the same seen before, but with biases
        float gene_value = biases[idx_model_param_gen];

        if (curand_uniform(&state) < mutation_prob) {
            float delta = (curand_uniform(&state) * 2.0f - 1.0f) * mutation_range;
            gene_value += delta;
        }

        int idx_model_param_out = (output_idx * num_bias_per_model) + idx - num_weights_per_model;
        new_biases[idx_model_param_out] = gene_value;

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
    int raggio,                                 
    float* input_workspace_addr,                
    cudaStream_t stream
){

    int n_thread = 32;
    int n_block = (raggio + n_thread -1) / n_thread;
    dim3 threads(n_thread,n_thread);
    dim3 blocks(n_block,n_block);
    vision_kernel<<<blocks,threads,0,stream>>>(
        world_value,
        world_id,
        world_signaling,
        dim_world,
        cell_idx,
        raggio,
        input_workspace_addr
    );
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("errori vision_kernel: %s\n",cudaGetErrorString(cudaGetLastError()));
              


}

// ===================================================================================================

// Wrapper kernel NN_forward
// compute the output of each layer sequntialy
void launch_NN_forward(                                     
    float* input_workspace_addr,                    
    float* output_workspace_addr,                   
    float* weights,                                 
    int n_weights,                                  
    float* biases,                                  
    int n_biases,                                   
    int* structure,                                 
    int cell_index,                                 
    int *cells,                                     
    int *world_id,                                  
    int dim_structure,                              
    cudaStream_t stream    
){
    int n_thread_per_block = 1024;
    int layer1_size = 0;
    int layer2_size = 0;
    int structureLenght = dim_structure;

    int weight_offset = 0;
    int biases_offset = 0;

    for(int i=0; i < (structureLenght-1); i++){
        layer1_size = structure[i];
        layer2_size = structure[i + 1];

        int thread_number = layer1_size * layer2_size;             

        int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;

        if(i == (structureLenght-2)){
            NN_forward_kernel<<<n_block, n_thread_per_block, 0 , stream>>>(
                input_workspace_addr,
                output_workspace_addr, 
                weights, 
                biases, 
                cell_index, 
                cells, 
                world_id, 
                n_weights, 
                n_biases, 
                layer1_size, 
                layer2_size, 
                weight_offset, 
                biases_offset
            );  
            if(cudaGetLastError()!=cudaError::cudaSuccess) printf("errori NN_forward_kernel_O: %s\n",cudaGetErrorString(cudaGetLastError()));
                
        }
        else{
            
            NN_forward_kernel<<<n_block, n_thread_per_block, 0 , stream>>>(
                input_workspace_addr,
                input_workspace_addr, 
                weights, 
                biases, 
                cell_index, 
                cells, 
                world_id, 
                n_weights, 
                n_biases, 
                layer1_size, 
                layer2_size, 
                weight_offset, 
                biases_offset
            );  
            if(cudaGetLastError()!=cudaError::cudaSuccess) printf("errori NN_forward_kernel: %s\n",cudaGetErrorString(cudaGetLastError()));
    
            
        }

        // update offset
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
    float* output_workspace_addr,                  
    int* cells,                                     
    int world_dim,                                  
    int number_of_creatures,                        
    int output_size,                               
    int cell_index,                                 
    float energy_fraction,
    cudaStream_t stream
){
    int n_thread_per_block = 1024;
    int thread_number = output_size;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;

    output_elaboration_kernel<<<n_block, n_thread_per_block, 0, stream>>>(
        world_value,
        world_signal,
        world_id,
        contribution_matrix,
        output_workspace_addr,
        cells,
        world_dim,
        number_of_creatures,
        output_size,
        cell_index,
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

    compute_energy_and_occupation_kernel<<<n_block,n_thread_per_block,0,stream>>>(
        world_value,
        world_id,
        occupation_vector,
        energy_vector,
        world_dim,
        n_creature);


}

// ==================================================================================================

// Wrapper: recombine_model
void launch_recombine_models_kernel(
    float *d_weights, float *d_biases,
    float *d_new_weights, float *d_new_biases,
    int num_weights_per_model, int num_bias_per_model,
    int model1_idx, int model2_idx, int output_idx,
    float gen_x_block,
    float mutation_prob,
    float mutation_range,
    unsigned long seed,
    cudaStream_t stream) 
{
    // totla number of genes (weights + biases)
    int total_genes = num_weights_per_model + num_bias_per_model;

    int threads_per_block = gen_x_block*total_genes +1;
    if(threads_per_block>1024) threads_per_block = 1024;
    int num_blocks = (total_genes + threads_per_block - 1) / threads_per_block;

    recombine_models_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_weights,
        d_biases,
        d_new_weights,
        d_new_biases,
        num_weights_per_model,
        num_bias_per_model,
        model1_idx,
        model2_idx,
        output_idx,
        mutation_prob,
        mutation_range,
        seed
    );

}