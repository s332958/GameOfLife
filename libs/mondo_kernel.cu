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
        // calcolo indice thread       
        int center_x = blockIdx.x*blockDim.x + threadIdx.x;    
        int center_y = blockIdx.y*blockDim.y + threadIdx.y;
        int index = center_y * dim_world + center_x;
        
        // se i thread superano la dim del mondo ritorno
        if(center_x >= dim_world || center_y >= dim_world) return;
        
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
    
    //==============================================================================================
    
__global__ void cellule_cleanup_kernel(int *cellule_cu, int* temp_cellule_cu, int *id_matrix, int* cellCount, int* mask_cu, bool* mask_alive){
        
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        //ritorno i thread che eccedono il numero di cellule vive
        if(idx >= *cellCount) return;

        //inizializzo la maschera a 0 e temp cellule a 0 e alive a false
        mask_cu[idx] = 0;
        temp_cellule_cu[idx] = 0;
        mask_alive[idx] = false;
    
        // se trovo cellule con id > 0 quindi occupate da creature segno 1 in mask_cu e true in mask_alive
        if(id_matrix[cellule_cu[idx]] > 0){
           mask_cu[idx] = 1; 
           mask_alive[idx] = true;
        }  
        //printf("ID_MATRIX: %d IDX: %d T: %d cell count: %d\n",id_matrix[cellule_cu[idx]],cellule_cu[idx],idx,*cellCount);
        // dichiaro un valore di parallelizzazione 32 è ottimale poicheè è la dimensione di un warp
        int dim_paral = 32;
        int id_sort_x = idx % dim_paral;
        int id_sort_y = idx / dim_paral;
        
        // indice di ordinamento
        int index_sort = id_sort_y * dim_paral + id_sort_x;
        
        // tutti i thread_x con 0 iniziano a fare parallelizzazione
        if (id_sort_x == 0){
            int increment = 0;
            for (int i = 0; i < dim_paral; i++){
                
                if(index_sort + i < *cellCount){ 
                    increment = increment + mask_cu[index_sort + i];               
                    mask_cu[index_sort + i] = increment;
                }

            }    
        }    
        __syncthreads();
    
        if (id_sort_y != 0){
            mask_cu[index_sort] = mask_cu[index_sort] + mask_cu[id_sort_y*dim_paral - 1];
        }    
        __syncthreads();
    
        if (mask_alive[idx] == true){
            temp_cellule_cu[mask_cu[idx] - 1] = cellule_cu[idx];
        }     
        int new_cellCount = mask_cu[*cellCount - 1];
        if (idx > new_cellCount)return;
    
        cellule_cu[idx] = temp_cellule_cu[idx];  
    
        if (idx != 0) return;
        *cellCount = new_cellCount; 
    
        
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
    int thread_per_dimension = sqrt(n_thread_per_block);
    int n_block = world_dim / thread_per_dimension;
    if(n_block==0) n_block = 1;
    dim3 thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    dim3 block_number = dim3(n_block, n_block); 

    world_update_kernel<<<block_number,thread_number,0,stream>>>(
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

//Wrapper cellule cleanup
void launch_cellule_cleanup(int* cells, int* cellCount_h, int* cellCount_d, int* id_matrix, cudaStream_t stream){

    int cellCountR = *cellCount_h;
    if(cellCountR == 0) {
        // printf("cellCount = 0 \n");
        return;
    }
    
    int* temp_cellule_cu;
    int* mask_cu;
    bool* mask_alive;
    cudaMalloc((void**)&mask_alive, cellCountR * sizeof(bool));
    cudaMalloc((void**)&mask_cu, cellCountR * sizeof(int));
    cudaMalloc((void**)&temp_cellule_cu, cellCountR * sizeof(int));

    cudaMemset(mask_alive,0,cellCountR * sizeof(bool));
    cudaMemset(mask_cu,0,cellCountR * sizeof(int));
    cudaMemset(temp_cellule_cu,0,cellCountR * sizeof(int));

    int n_thread_per_block = 1024; //properties.maxThreadsPerBlock; 
    int thread_number = cellCountR;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;

    cellule_cleanup_kernel<<<n_block, n_thread_per_block, 0, stream>>>(cells, temp_cellule_cu, id_matrix, cellCount_d, mask_cu, mask_alive);
    
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("errori cellule_cleanup_kernel: %s\n",cudaGetErrorString(cudaGetLastError()));
    cudaFree(temp_cellule_cu);
    cudaFree(mask_alive);
    cudaFree(mask_cu);
}
