#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include "mondo_kernel.cuh"

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <random>

// =========================================================================================================

__global__ void add_objects_to_world_kernel(float *world_value, int *world_id, int dim_world, 
                                    int id, float min_value, float max_value, float threashold
                                ){
    
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int idx = x + y*dim_world;

    if(idx<dim_world*dim_world){

        if(world_id[idx]==0){
            curandState state;
            curand_init(clock64(),threadIdx.x,0,&state);
            float p_occupation = curand_uniform(&state);

            if(p_occupation>threashold){
                float value = curand_uniform(&state)*(max_value - min_value) + (min_value);
                world_id[idx] = id;
                world_value[idx] = value;
            }

        }
    
    }

}

// =========================================================================================================

__global__ void world_update_kernel(
    float *world_value, 
    int *id_matrix, 
    float *contribution_matrix,
    int dim_world, 
    int number_of_creatures, 
    int *cellCount, 
    int *cells
)
    {                 
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        
        // se i thread superano la dim del mondo ritorno
        if(index >= dim_world*dim_world) return;
        
        // trovo ID e valore cella data la matrice degli id e indice cella mondo
        int ID = id_matrix[index];
        float starting_value = world_value[index];
        
        // se l'ID è -1 ovvero un ostacolo ritorno
        if(ID == -1) return;
        
        // setto i valori finale e id finale i valori iniziali della cella
        float final_value = starting_value;
        int final_id = ID;
        
        // eseguo un for con dimensione pari al numero massimo di creature per trovare la creatura che ha contribuito di piu 
        int max_id = 0;
        float max_value = 0;
        float ally_energy = 0;
        float enemy_energy = 0;
        
        for(int i = 0; i < number_of_creatures; i++){
            // prendo il valore di contributo dato ID e indice cella mondo
            float value = contribution_matrix[i * dim_world * dim_world + index]; 
            if (ID == i+1){
                // se l'id del contributo è uguale all'id della cella allora aumenta l'energia alleata
                ally_energy += value;      
            }    
            else{
                // se l'id del contributo è diverso dall'id della cella corrente allora aumenta l'energia nemica
                enemy_energy += value;
            }
            // mi salvo il massimo cotribuente e il suo valore                      
            if (value > max_value){
                max_value = value;
                max_id = i+1;
            }    
        } 

        // eseguo i conti se la cella iniziale era libera id=0 
        //printf("Thread %d ID %d says %f, %f!\n", index, ID, ally_energy, enemy_energy);       
        if (ID == 0){
            if (enemy_energy > 0){
                
                

                // calcolo il valore finale come somma pesata del valore iniziale per il contributo, piu la somma del massimo contribuente 
                // assegno l'id al contribuente maggiore
                final_value = starting_value * (max_value / enemy_energy) + max_value;
                final_id = max_id;
                
                // aggiorno il numero di celle vive e salvo l'indice
                int pos = atomicAdd(cellCount, 1);
                cells[pos] = index;
                //printf("UPDATE CELL ALIVE: %d con index %d \n",pos-1,index);
            }    
        } 
        
        // eseguo i conti se la cella iniziale era occupata id>0
        else{
            final_value = final_value - 0.05;
            if (starting_value + ally_energy - enemy_energy < 0){
                // se la cella è occupata e la forza delle celle nemiche supera quella corrente + alleate allora la cella muore e l'eccesso viene lasciato come cibo
                final_value = abs(starting_value + ally_energy - enemy_energy);
                final_id = 0;
            }            
            else{
                // se la cella è occupata ma la forza nemica è inferiore ad alleati + corrente si calcola solo la somma tra alleata corrente e - nemici e questo è il nuovo risultato
                final_value = starting_value + ally_energy - enemy_energy;
            }    
        }    
        
        // se il valore finale ha una soglia troppo bassa allora l'energia va al mondo
        if(final_value < 0.02f){
           final_id = 0;
        }     
        
        // se il valore finale ha una soglia troppo alta allora l'energia viene impostata al max 1
        if(final_value > 1.0f){
            final_value = 1.0f;
        }    

        // assegno i valori finali alla cella in analisi
        world_value[index] = final_value;                   
        id_matrix[index] = final_id; 

}    
    
    //================================================================================
    
    //Wrapper add objects to world
void launch_add_objects_to_world(float* world_value_d, int* world_id_d, int dim_world,
                                int id, float min_value, float max_value, float threshold,
                                cudaStream_t stream) {

    dim3 blockDim(16, 16);
    dim3 gridDim((dim_world + 15) / 16, (dim_world + 15) / 16);

    add_objects_to_world_kernel<<<gridDim, blockDim, 0, stream>>>(
        world_value_d, world_id_d, dim_world,
        id, min_value, max_value, threshold
    );

}

//Wrapper mondo_cu update
void launch_world_update(
    float *world_value,
    int *id_matrix,
    float *contribution_matrix, 
    int *cells,
    int world_dim, 
    int number_of_creatures,
    int *cellCount, 
    cudaStream_t stream
){


    int n_thread_per_block = 1024; 
    int thread_number = world_dim*world_dim;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;

    world_update_kernel<<<n_block,n_thread_per_block,0,stream>>>(
        world_value, 
        id_matrix,
        contribution_matrix,
        world_dim,
        number_of_creatures,
        cellCount,
        cells
    );
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("errori world_update_kernel: %s\n",cudaGetErrorString(cudaGetLastError()));
    

}





















































// ===============================================================================================================================================================
// NUOVO TEST


__global__ void find_index_cell_alive_kernel(
    int *world_id,
    int *cell_alive_vector,
    int world_dim_tot,
    int *n_cell_alive
) {
    extern __shared__ int shared_mem[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    int is_alive = 0;

    if (idx < world_dim_tot) {
        is_alive = (world_id[idx] > 0);
        cell_alive_vector[idx] = is_alive * (idx+1);
    }

    // Scriviamo il valore nella shared memory (0 se fuori dai limiti)
    shared_mem[tid] = (idx < world_dim_tot) ? is_alive : 0;
    __syncthreads();

    // Riduzione generale per blocchi NON potenze di due
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s+1 && tid+s+1 <blockDim.x) {
            shared_mem[tid] += shared_mem[tid + s+1];
            shared_mem[tid + s+1]=0;
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_mem[0] += shared_mem[1];
        shared_mem[1] = 0;
        atomicAdd(n_cell_alive, shared_mem[0]);
    }
}

__global__ void compact_cell_alive_kernel_pt1(
    int *alive_cell_vector,
    int *support_vector,
    int *n_alive_cell,
    int world_dim
) {
    extern __shared__ int shared_mem[]; // [2 * blockDim.x + 1]

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    // Clear shared memory
    shared_mem[local_idx] = 0;
    shared_mem[blockDim.x + local_idx] = 0;

    // Carica in shared
    if (global_idx < world_dim) {
        shared_mem[local_idx] = alive_cell_vector[global_idx];
        alive_cell_vector[global_idx] = 0;
    }

    __syncthreads();

    // Filtro: fatto da thread 0
    if (local_idx == 0) {
        int count = 0;
        for (int i = 0; i < blockDim.x; i++) {
            int val = shared_mem[i];
            if (val > 0) {
                shared_mem[blockDim.x + count] = val;
                count++;
            }
        }

        shared_mem[2 * blockDim.x] = count;
        support_vector[blockIdx.x] = count;
    }

    __syncthreads();

    // Scrittura valori filtrati nel buffer alive (solo entro limiti)
    int count = shared_mem[2 * blockDim.x];
    if (local_idx < count) {
        alive_cell_vector[global_idx] = shared_mem[blockDim.x + local_idx];
    }
}



__global__ void compact_cell_alive_kernel_pt2(int *alive_cell_vector, int *support_vector, int *n_alive_cell, int n_block, int dim_block){

    __shared__ int shared_mem[2];       
    //mem[0] starting index, mem[1] number of element 

    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(n_block==0 && threadIdx.x==0){
        shared_mem[0] = 0;
        shared_mem[1] = support_vector[n_block];
        support_vector[n_block] = shared_mem[0] + shared_mem[1];
    }else if(n_block>0 && threadIdx.x==0){
        shared_mem[0] = support_vector[n_block-1];
        shared_mem[1] = support_vector[n_block];
        support_vector[n_block] = shared_mem[0] + shared_mem[1];
    }

    __syncthreads();

    int idx_alive_cell_read = n_block*dim_block+idx;
    if(idx<shared_mem[1]){
        int offset = shared_mem[0];
        //printf("read cell: %d \n",idx_alive_cell_read);
        int idx_alive_cell_write = offset+idx;
        alive_cell_vector[idx_alive_cell_write] = alive_cell_vector[idx_alive_cell_read]-1;
    }


}


//Wrapper compute alive cell
void launch_find_index_cell_alive(
    int *world_id,
    int world_dim_tot,
    int *alive_cell_vector,
    int *n_cell_alive_d,
    int *n_cell_alive_h,
    cudaStream_t stream
) {
    int n_thread = 1024;
    if(world_dim_tot<n_thread) n_thread = world_dim_tot;
    int n_block = (world_dim_tot+n_thread-1) / n_thread;

    int *support_vector;
    cudaMalloc((void**) &support_vector, world_dim_tot*sizeof(int));

    cudaMemsetAsync(n_cell_alive_d, 0, sizeof(int),stream);

    find_index_cell_alive_kernel<<<n_block,n_thread,sizeof(int)*(n_thread+1),stream>>>(
        world_id,
        alive_cell_vector,
        world_dim_tot,
        n_cell_alive_d
    );

   cudaDeviceSynchronize();

    n_thread = 512;
    n_block = (world_dim_tot+n_thread-1) / n_thread;

    compact_cell_alive_kernel_pt1<<<n_block,n_thread,sizeof(int)*(n_thread*2+1),stream>>>(
        alive_cell_vector,
        support_vector,
        n_cell_alive_d,
        world_dim_tot
    );

    int block_dim_pt1 = n_thread;
    int n_block_pt1 = n_block;

    for(int i=0; i<n_block_pt1; i++){
        compact_cell_alive_kernel_pt2<<<1,n_thread,0,stream>>>(
            alive_cell_vector,
            support_vector,
            n_cell_alive_d,
            i,
            block_dim_pt1
        );
    }

    cudaFree(support_vector);


}