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
    int raggio,
    int input_workspace,
    float* input) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x; // colonna nella finestra raggio x raggio
    int y = threadIdx.y + blockIdx.y * blockDim.y; // riga nella finestra raggio x raggio

    if (x >= raggio || y >= raggio) return;

    int center_row = *cell_idx / dim_world;
    int center_col = *cell_idx % dim_world;
    //printf("%d\n",*cell_idx);

    // Calcola coordinate "virtuali" senza wrapping
    int world_row = (center_row - (raggio / 2) + y + dim_world) % dim_world;
    int world_col = (center_col - (raggio / 2) + x + dim_world) % dim_world;

    int offset = y*raggio + x;
    int workspace_offset = input_workspace*raggio*raggio*3;

    int world_pos = world_row * dim_world + world_col;
    input[workspace_offset + offset + 0] = world_value[world_pos];
    input[workspace_offset + offset + 1] = static_cast<float>(world_id[world_pos]);
    input[workspace_offset + offset + 2] = world_signaling[world_pos];
}

// ============================================================================


__global__ void NN_forward_kernel(
    float* input,
    float* output, 
    float* weights, 
    float* biases, 
    int cellule_index, 
    int* cellule,
    int* matrice_id, 
    int n_weights, 
    int n_biases, 
    int layer1_size, 
    int layer2_size,
    int layerInput_size
    int layerOutput_size){
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index > layer2_size * layer1_size){
            return;
        }

        int mondo_index = cellule[cellule_index];
        int ID = matrice_id[mondo_index];

        int weight_index = n_weights * (ID - 1) + index;
        
        int output_index = index / layer1_size + cellule_index * layerOutput_size; 
        int input_index = index % layer1_size + cellule_index * layerInput_size;
        
        float weighted = weights[weight_index] * input[input_index];

        __syncthreads();

        if (index < layer2_size){
            output[index + cellule_index * layerInput_size] = 0;
        }
        
        atomicAdd(&output[output_index], weighted);
        
        if (index >= layer2_size){
            return;
        }
        
        int bias_index = n_biases * (ID - 1) + index;

        output[index] += biases[bias_index];

        output[index] = relu(output[index]);

}

//===================================================================================


__global__ void output_elaboration_kernel(
    float* mondo,
    float* mondo_signal,
    float* mondo_contributi,
    int* id_matrix,
    int number_of_creatures,
    int dim_mondo,
    float* output,    
    int output_size,
    int* cellule,
    int offset,
    int num_work
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index > output_size * num_work)return;

    float output = output[index];

    output = sigmoid(output);
    


    int center_index = cellule[index / output_size + offset];
    int output_index = index % output_size;
    int ID = id_matrix[center_index];


    if(output_index == output_size-1){
        float signal = (1/number_of_creatures)*(ID-1)+(output/number_of_creatures);
        mondo_signal[center_index] = signal;
        return;
    }
    float center_value = mondo[center_index];

    int center_x = center_index % dim_mondo;
    int center_y = center_index / dim_mondo;

    int dim_out_window = sqrtf(output_size-1);
    int raggio = dim_out_window/2;
    int filter_x = (output_index % dim_out_window - raggio + center_x)%dim_mondo;
    int filter_y = (output_index / dim_out_window - raggio + center_y)%dim_mondo;
    
    int filter_index = dim_mondo * dim_mondo * (ID - 1) + filter_y * dim_mondo + filter_x;

    float frazione_di_se_stesso = 0.05;

    float final_output = center_value * frazione_di_se_stesso * output;

    mondo_contributi[filter_index] = final_output;

    atomicAdd(&mondo[center_index], - final_output);    
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
    int input_workspace,
    float* input,
    cudaStream_t stream
){

    int n_thread = 32;
    int n_block = (raggio + n_thread -1) / n_thread;
    dim3 threads(n_thread,n_thread);
    dim3 blocks(n_block,n_block);
    vision_kernel<<<blocks,threads,0,stream>>>(world_value,world_id,world_signaling,dim_world,cell_idx,raggio,input_workspace,input);


}

// Wrapper kernel NN_forward
void launch_NN_forward(
    float* input,
    float* output,
    float* weights,
    int n_weights,
    float* biases,
    int n_biases,
    int* structure,
    int cellule_index,
    int *cellule,
    int* matrice_id    
){
    int n_thread_per_block = properties.maxThreadsPerBlock;
    int layer1_size = 0;
    int layer2_size = 0;
    int structureLenght = sizeof(structure) / sizeof(structure[0]);
    int layerInput_size = structure[0];
    int layerOutput_size = structure[structureLenght];
    int weight_offset = 0;
    int bias_offset = 0;
    for(int i, i < (structureLenght-1), i++){
        layer1_size = structure[i];
        layer2_size = structure[i + 1];
        weight_offset = //da aggiornare
        bias_offset = //da aggiornare

        thread_number = layer1_size * layer2_size;        
        int n_block = thread_number / n_thread_per_block;
        if(n_block%n_thread_per_block!=0 || n_block==0)n_block=n_block+1;      

        if(i == (structureLenght-1)){
            NN_forward_kernel<<<n_block, n_thread_per_block>>>(input, output, &weights[weight_offset], &biases[bias_offset], cellule_index, cellule, matrice_id, n_weights, n_biases, layer1_size, layer2_size, layerInput_size, layerOutput_size);   
        }
        else{
            NN_forward_kernel<<<n_block, n_thread_per_block>>>(input, input, &weights[weight_offset], &biases[bias_offset], cellule_index, cellule, matrice_id, n_weights, n_biases, layer1_size, layer2_size, layerInput_size, layerOutput_size);    
        }
    }

}

//Wrapper kernel output_elaboration
void launch_output_elaboration(
    float* mondo_signal,
    float* mondo_contributi,
    int* id_matrix,
    int number_of_creatures,
    int dim_mondo,
    float* output,
    int output_size,
    int* cellule,
    int offset,
    int num_work
){
    int n_thread_per_block = properties.maxThreadsPerBlock;
    thread_number = output_size * num_work;
    int n_block = thread_number / n_thread_per_block;
    if(n_block%n_thread_per_block!=0 || n_block==0)n_block=n_block+1; 

    output_elaboration_kernel<<<n_block, n_thread_per_block>>>(mondo_signal, mondo_contributi, id_matrix, number_of_creatures, dim_mondo, output, output_size, cellule, offset, num_work);
}