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

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index_cell = index / dim_input;
    int vision_index = index % dim_input;
    
    if (index >= dim_input*limit_workspace_cell) return;

    int dim_window_sq = dim_window*dim_window;
    int radius = dim_window / 2;

    int center_index = cell_idx[index_cell];

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
        int cell_index = tidx / (layer1_size * layer2_size);
        int weight_index = tidx % (layer1_size * layer2_size);
       
        int world_index = cells[cell_index];
        int ID = world_id[world_index];
                
        int true_weight_index = n_weights * (ID - 1) + weight_index + offset_weights; // + offset pesi siccome dipendono dal layer (es layer0 offeset=0, layer1 offset=layer0*layer1)
         
        int input_neuron_idx  = weight_index % layer1_size;
        int output_neuron_idx = weight_index / layer1_size;

        int input_index  = cell_index * layer1_size  + input_neuron_idx;
        int output_index = cell_index * layer2_size  + output_neuron_idx;
            
        float weighted = weights[true_weight_index] * input[input_index];
      
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
        // printf("thread: %4d, bias: %.4f totale_dopo: %4.4f \n",tidx,biases[bias_index],output_addr[tidx]);

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

    float final_output = center_value * energy_fraction * output;
    int dim_window_sq = dim_window*dim_window;
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
    int n_creature) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= world_dim * world_dim) return;

    int id = world_id[index] - 1;

    if (id < 0)return;
    
    atomicAdd(&occupation_vector[id], 1.0f/float(world_dim));
    atomicAdd(&energy_vector[id], world_value[index]);

}

// ==================================================================================

// PREMESSA:
// la struttura dei blocchi è importante, il numero di thread dentro ad un blocco, indica il numero di pesi/biases che ogni blocco genetico possiede
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

    // Calcolo indice thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_genes = num_weights_per_model + num_bias_per_model;
    // faccio una return dei thread che superano la somma di pesi+biases
    if (idx >= total_genes) return;

    // creazione d curandstate per generare valori casuali su device
    curandState state;
    curand_init(seed, idx, 0, &state);

    // il primo thread di ogni blocco si occupa di generare il numero del genitore del blocco genetico
    if(threadIdx.x==0){
        int gen = curand(&state) % 2;
        gen_id = gen==0?model1_idx:model2_idx;
    }

    __syncthreads();

    int idx_model_param_gen = -1;

    // caso in cui stiamo analizzando i pesi (indice < numero pesi modello)
    if (idx < num_weights_per_model) {

        // calcolo l'indice dei geni (pesi singoli) dal blocco genetico del genitore
        idx_model_param_gen = (gen_id * num_weights_per_model) + idx;

        // carico il valore del gene 
        float gene_value = weights[idx_model_param_gen];

        // genero un numero casuale per definire se il gene subisce una mutazione (caso per cui viene superata la soglia di mutazione)
        if (curand_uniform(&state) < mutation_prob) {
            // calcolo il delta della variazione che va da -mutation_range a mutation_range
            float delta = (curand_uniform(&state) * 2.0f - 1.0f) * mutation_range;
            // applico il delta
            gene_value += delta;
        }

        // trovo l'indice per scrivere il nuovo valore sul nuovo modello e lo aggiorno
        int idx_model_param_out = (output_idx * num_weights_per_model + idx);
        new_weights[idx_model_param_out] = gene_value;

    }else{

        //caso in cui siamo nei bias (indice > numero peso modelli) 
        //calcolo indice gene genitore come per i pesi ma si toglie l'offset del numero di pesi (questo perche i thread dei biases sono tutti dopo i pesi)
        idx_model_param_gen = (gen_id * num_bias_per_model) + idx - num_weights_per_model;

        // carico il valore del gene in un registro
        float gene_value = biases[idx_model_param_gen];

        // genero un valore casuale che se supera la soglia allora indica la mutazione del gene
        if (curand_uniform(&state) < mutation_prob) {
            // calcolo il valore di mutzione del gene come fatto in precedenza 
            float delta = (curand_uniform(&state) * 2.0f - 1.0f) * mutation_range;
            // aggiorno il valore del nuovo gene
            gene_value += delta;
        }

        // trovo l'indice su dove va scritto il bias appena calcolato e lo aggiorno
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
    //if(cudaGetLastError()!=cudaError::cudaSuccess) printf("errori vision_kernel: %s\n",cudaGetErrorString(cudaGetLastError()));          


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
        //if(cudaGetLastError()!=cudaError::cudaSuccess) printf("errori NN_forward_kernel: %s\n",cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(input_workspace, output_workspace, workspace_size*limit_workspace_cell, cudaMemcpyDeviceToDevice);
        
        if (i < (dim_structure-2)){
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
    //if(cudaGetLastError()!=cudaError::cudaSuccess) printf("errori output_elaboration_kernel: %s\n",cudaGetErrorString(cudaGetLastError()));
    

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
    // Numero totale di geni (pesi + bias)
    int total_genes = num_weights_per_model + num_bias_per_model;

    // Imposta configurazione kernel
    int threads_per_block = gen_x_block*total_genes +1;
    if(threads_per_block>1024) threads_per_block = 1024;
    int num_blocks = (total_genes + threads_per_block - 1) / threads_per_block;

    // Lancia il kernel
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