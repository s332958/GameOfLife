__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ float fast_sigmoid(float x) {
    return 0.5f * (x / (1.0f + fabsf(x))) + 0.5f;  
}



__global__ void kernel_NN_forward(
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


void NN_forward(
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

        thread_number = layer1_size * layer2_size;        
        int n_block = thread_number / n_thread_per_block;
        if(n_block%n_thread_per_block!=0 || n_block==0)n_block=n_block+1;      

        if(i == (structureLenght-1)){
            kernel_NN_forward<<<n_block, n_thread_per_block>>>(input, output, &weights[weight_offset], &biases[bias_offset], cellule_index, cellule, matrice_id, n_weights, n_biases, layer1_size, layer2_size, layerInput_size, layerOutput_size);   
        }
        else{
            kernel_NN_forward<<<n_block, n_thread_per_block>>>(input, input, &weights[weight_offset], &biases[bias_offset], cellule_index, cellule, matrice_id, n_weights, n_biases, layer1_size, layer2_size, layerInput_size, layerOutput_size);    
        }
    }

}

__global__ void kernel_output_elaboration(
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

    int frazione_di_se_stesso = 0.08

    float final_output = center_value * frazione_di_se_stesso * output;

    mondo_contributi[filter_index] = final_output;

    atomicAdd(&mondo[center_index], - final_output);
    
}



void output_elaboration(
    float* mondo_signal,
    float* mondo_contributi,
    int* id_matrix
    int number_of_creatures,
    int dim_mondo,
    float* output,
    int output_size,
    int* cellule,
    int offset,
    int num_work
){
    int n_thread_per_block = properties.maxThreadsPerBlock;
    thread_number = output_size * numCellule;
    int n_block = thread_number / n_thread_per_block;
    if(n_block%n_thread_per_block!=0 || n_block==0)n_block=n_block+1; 

    kernel_output_elaboration<<<n_block, n_thread_per_block>>>(mondo_signal, mondo_contributi, id_matrix, number_of_creatures, dim_mondo, output, output_size, cellule, offset, num_work);
}