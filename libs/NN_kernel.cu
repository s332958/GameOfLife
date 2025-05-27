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
    int raggio,
    float* workspace_addr
) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x; // colonna nella finestra raggio x raggio
    int y = threadIdx.y + blockIdx.y * blockDim.y; // riga nella finestra raggio x raggio

    if (x >= raggio || y >= raggio) return;

    int center_row = *cell_idx / dim_world;
    int center_col = *cell_idx % dim_world;

    // Calcola coordinate "virtuali" senza wrapping
    int world_row = (center_row - (raggio / 2) + y + dim_world) % dim_world;
    int world_col = (center_col - (raggio / 2) + x + dim_world) % dim_world;

    int offset = (y*raggio + x)*2;
    //printf("thread: %d \n",offset);

    int world_pos = world_row * dim_world + world_col;
    workspace_addr[offset + 0] = world_value[world_pos];
    workspace_addr[offset + 1] = world_signaling[world_pos];
}

// ============================================================================


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
        // index del thread 
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;

        // se il thread supera la dim dei weights  allora ritorno 
        if (tidx >= layer2_size * layer1_size){
            return;
        }

        // ottengo l'indice di dove si trova la cella viva cosi poi accedo al suo ID per capire che modello usare
        int world_index = cells[cell_index];
        int ID = world_id[world_index];
        //printf("thread: %d ID: %d index: %d \n",tidx, ID, world_index);

        // ottengo l'accesso ai pesi del modello corretto 
        int weight_index = n_weights * (ID - 1) + tidx + offset_weights; // + offset pesi siccome dipendono dal layer (es layer0 offeset=0, layer1 offset=layer0*layer1)
        
        // assegno ad ogni thread la cella da cui leggere l'input e dove assegnare il suo output
        int output_index = tidx % layer2_size;
        int input_index = tidx / layer2_size;
        
        // ottengo per ogni thread il suo input*weight
        float weighted = weights[weight_index] * input_addr[input_index];

        // pulisco le celle di memoria che mi servono in output
        if (tidx < layer2_size){
            output_addr[tidx] = 0;
        }
        /*
        
        printf("thread in azione: %4d, ID: %4d, weight: %4d, output_idx: %4d, input_idx: %4d\n",tidx, ID, weight_index, output_index, input_index);
        printf("thread: %4d, input: %4.4f, weight: %4.4f, input_weight: %4.4f output: %d\n",tidx,input_addr[input_index],weights[weight_index],weighted,output_index);
        */

        __syncthreads();

        // sommo per ogni cella di output i propri pesi
        atomicAdd(&output_addr[output_index], weighted);
        
        // blocco tutti i thread che hanno id maggiore della dimensione dei bias del layer di output
        if (tidx >= layer2_size){
            return;
        }
        
        // otttengo i bias realitivi in base al modello corretto
        int bias_index = n_biases * (ID - 1) + tidx + offset_biases;  // + offset pesi siccome dipendono dal layer (es layer1 offeset=0, layer2 offset=layer1)

        __syncthreads();

        //printf("thread in azione: %4d, ID: %4d, bias_index: %4d\n",tidx, ID, bias_index);

        // sommo il bias alla cella di output corretta
        output_addr[tidx] += biases[bias_index];
        // printf("thread: %4d, bias: %.4f totale_dopo: %4.4f \n",tidx,biases[bias_index],output_addr[tidx]);

        // applico la relu ad ogni cella di output modificata 
        output_addr[tidx] = relu(output_addr[tidx]);

}

//===================================================================================


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
    int cell_index
){  
    //indice del thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    //se eccedo il numero di output blocco il thread
    if(index >= output_size) return;

    //prendo il valore calcolato nell'output
    float output = outputs[index];

    //applico sigmoid
    output = sigmoid(output);

    //trovo l'indice della cella del mondo per trovare il suo ID, nel mentre trovo i punti dove scrivere l'output
    int center_index = cells[cell_index];
    int output_index = index;
    int ID = world_id[center_index];

    //l'ultimo thread va a modificare il mondo di signaling 
    if(output_index == output_size-1){
        float signal = (1/number_of_creatures)*(ID-1)+(output/number_of_creatures);
        world_signal[center_index] = signal;
        return;
    }

    // prendo tramite indice cella, il valore corrispettivo nel mondo
    float center_value = world_value[center_index];

    // calcolo la x e la y dell'indice della cella del mondo
    int center_x = center_index % world_dim;
    int center_y = center_index / world_dim;

    //trovo il raggio di intorno delle celle che vengono modificate della contribution matrix
    int dim_out_window = sqrtf(output_size-1);
    int raggio = dim_out_window/2;

    // trovo le corrispettive x e y di ogni cella modificata (considerando il mondo toroidale)
    int filter_x = ((output_index % dim_out_window) - raggio + center_x + world_dim)%world_dim;
    int filter_y = ((output_index / dim_out_window) - raggio + center_y + world_dim)%world_dim;
    
    // trovo l'indice nel mondo della cella modificata
    int filter_index = (world_dim * world_dim * (ID - 1)) + (filter_y * world_dim) + filter_x;

    // indico un max di quanto ogni cella puo donare del suo valore max
    float frazione_di_se_stesso = 1.0/9.0;

    // calcolo il valore finale da mettere nella cella dei contributi
    float final_output = center_value * frazione_di_se_stesso * output;

    //printf("thread: %d max: %d idx: %d  ID: %d\n",index,world_dim*world_dim*number_of_creatures,filter_index,ID);

    // modifico celle dei contributi circostanti in base al final_output calcolato prima
    atomicAdd(&contribution_matrix[filter_index], final_output);

    // modifico il valore della cella che ha generato lo spostamento
    atomicAdd(&world_value[center_index], - final_output);    
}

// ===================================================================================

__global__ void compute_energy_and_occupation_kernel(
    float* world_value,
    int* world_id,
    int* occupation_vector,
    float* energy_vector,
    int world_dim,
    int n_creature) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= world_dim || y >= world_dim) return;

    int idx = x + y*world_dim;
    
    // index on vectors valuation
    int id = world_id[idx] -1;

    if (id >= 0 && id < n_creature) {
        // Accesso atomico per evitare race condition
        atomicAdd(&occupation_vector[id], 1);
        atomicAdd(&energy_vector[id], world_value[idx]);
    }

}

// ==================================================================================

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
    if (idx >= total_genes) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    if(threadIdx.x==0){
        int gen = curand(&state) % 2;
        gen_id = gen==0?model1_idx:model2_idx;
    }

    __syncthreads();

    int idx_model_param_gen = -1;

    if (idx < num_weights_per_model) {

        idx_model_param_gen = (gen_id * num_weights_per_model) + idx;

        float gene_value = weights[idx_model_param_gen];

        if (curand_uniform(&state) > mutation_prob) {
            float delta = (curand_uniform(&state) * 2.0f - 1.0f) * mutation_range;
            gene_value += delta;
        }

        int idx_model_param_out = (output_idx * num_weights_per_model + idx);
        new_weights[idx_model_param_out] = gene_value;

    }else{

        idx_model_param_gen = (gen_id * num_bias_per_model) + idx - num_weights_per_model;

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
void launch_vision(                 // Testata e funzionante 
    float* world_value,                         // mondo che contiene i valori
    int* world_id,                              // mondo che contiene gli id
    float* world_signaling,                     // mondo che contiene i signaling
    int dim_world,                              // dimensione del mondo 
    int* cell_idx,                              // cella di memoria della cellula viva
    int raggio,                                 // raggio di visione della cellula 
    float* input_workspace_addr,                // indirizzo della stazione di input su cui scrivere il risultato (dim area di memoria = raggio*raggio*3)
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
// Limitazione i layer successivi devono avere dim dimnore del layer iniziale
// Usare una sola cella di memoria
void launch_NN_forward(                                     // Testata e funzionante
    float* input_workspace_addr,                    //inizio memoria zona di input
    float* output_workspace_addr,                   //inizio memoria zona di output
    float* weights,                                 //pesi di tutti i modelli
    int n_weights,                                  //pesi x modello
    float* biases,                                  //biases tutti i modelli
    int n_biases,                                   //biases x modello
    int* structure,                                 //vettore con le dimensioni dei layer di modelli (tutti i modelli hanno le stesse dim)
    int cell_index,                                 //indice della cellula viva che si usa per calcolare l'input
    int *cells,                                     //array delle cellule vive
    int *world_id,                                  //mondo che contiene gli id delle celle
    int dim_structure,                              //numero di layer del modello (tutti i modelli hanno lo stesso numero di layer)
    cudaStream_t stream    
){
    int n_thread_per_block = 1024;
    int layer1_size = 0;
    int layer2_size = 0;
    int structureLenght = dim_structure;
    // printf("STRUCTURE LENGHT: %d",structureLenght);
    // int layerInput_size = structure[0];
    // int layerOutput_size = structure[structureLenght];
    int weight_offset = 0;
    int biases_offset = 0;

    for(int i=0; i < (structureLenght-1); i++){
        layer1_size = structure[i];
        layer2_size = structure[i + 1];

        int thread_number = layer1_size * layer2_size;        
        int n_block = thread_number / n_thread_per_block;
        if(n_block%n_thread_per_block!=0 || n_block==0)n_block=n_block+1;      

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

        // aggiornamento offset
        weight_offset = layer1_size*layer2_size;
        biases_offset   = layer2_size;

    }

}

// ===================================================================================================

//Wrapper kernel output_elaboration
void launch_output_elaboration(                                 // Funziona e testata
    float* world_value,                             // matrice mondo valori
    float* world_signal,                            // matrice che contiene i signaling
    int* world_id,                                  // matrice mondo che contiene gli id per cella
    float* contribution_matrix,                     // matrice dei contributi di ogni creatura
    float* output_workspace_addr,                   // area di memoria dell'output del workspace
    int* cells,                                     // array cellule vive
    int world_dim,                                  // dimensione del mondo
    int number_of_creatures,                        // numero di creature massime nel mondo
    int output_size,                                // dimensione dell'output
    int cell_index,                                 // indice cella viva in analisi
    cudaStream_t stream
){
    int n_thread_per_block = 1024;
    int thread_number = output_size;
    int n_block = thread_number / n_thread_per_block;
    if(n_block%n_thread_per_block!=0 || n_block==0)n_block=n_block+1; 

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
        cell_index
    );
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("errori output_elaboration_kernel: %s\n",cudaGetErrorString(cudaGetLastError()));
    

}

// ===================================================================================================

// Wrapper compute energy and occupation for evaluation
void launch_compute_energy_and_occupation(
    float* world_value,
    int* world_id,
    int* occupation_vector,
    float* energy_vector,
    int world_dim,
    int n_creature,
    cudaStream_t stream
){

    int n_thread = 32;
    int n_block = (world_dim + n_thread -1) / n_thread;
    dim3 dim_block(n_thread,n_thread);
    dim3 dim_grid(n_block,n_block);

    compute_energy_and_occupation_kernel<<<dim_grid,dim_block,0,stream>>>(
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