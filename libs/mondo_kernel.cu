#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include "mondo_kernel.cuh"

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <random>

// =========================================================================================================

// kernel for adding object to world (like obstacles or food)
__global__ void add_objects_to_world_kernel(float *world_value, int *world_id, int dim_world, 
                                    int id, float min_value, float max_value, float threashold,
                                    curandState curandStates[]
                                ){
    
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int idx = x + y*dim_world;

    if(idx<dim_world*dim_world){

        if(world_id[idx]==0){
            // instantiate a random generator
            curandState state = curandStates[threadIdx.x];
            float p_occupation = curand_uniform(&state);

            // if the random value is over a threashold, then generate a random value for the cell and store the chosen ID
            if(p_occupation>threashold){
                float value = curand_uniform(&state)*(max_value - min_value) + (min_value);
                world_id[idx] = id;
                world_value[idx] = value;
            }

        }
    
    }

}

// =========================================================================================================

// kernel that from a contribution matrix (that rappresent the contribution of each creature for each cell)
// compute the final creature fo each cell
__global__ void world_update_kernel(
    float *world_value, 
    int *id_matrix, 
    float *world_signal,
    float *contribution_matrix,
    int dim_world, 
    int number_of_creatures, 
    float energy_decay
)
    {                 
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        
      // if thread are greater that world_Dim return
        if(index >= dim_world*dim_world) return;
        
        // get ID and value from id_matrix and world_value
        int ID = id_matrix[index];
        float starting_value = world_value[index];
        
        // if the cell is occupated from an obstacles set -1 value and return
        // FORSE SI PUO TOGLIERE IL -1
        if(ID == -1) {
            world_value[index] = -10;
            return;
        }
        
        // set the final cell value and id with the value of past step
        float final_value = starting_value;
        int final_id = ID;
        
        // use a for cicle to find the model that give more contribution
        int max_id = 0;
        int fighters = 0;
        float max_value = 0;
        float ally_energy = 0;
        float enemy_energy = 0;
        float soglia_morte = 0.04f;
        
        for(int i = 0; i < number_of_creatures; i++){
            // get the contribution value given id (i) and offset cell
            float value = contribution_matrix[i * dim_world * dim_world + index]; 
            if (value > 0){
                fighters += 1;
                             
                if (ID == i+1){
                    // if contribution is from  a creature with the same ID cell, so raise the allay energy
                    ally_energy += value;      
                }    
                else{
                    // if contribution is from a creature with different ID cell, raise enemy energy
                    enemy_energy += value;
                }
                // save the most creature that give more energy and the energy that it gives 
                if (value > max_value){
                    max_value = value;
                    max_id = i+1;
                } 
            }   
        } 
        if(ID > 0){
            float energy = starting_value + ally_energy - enemy_energy - energy_decay;
            
            if (energy < 0){
                final_value = 0;
                final_id = 0;
            }            
            else{
                final_value = energy;
            }

            if(final_value < soglia_morte){
                final_id = 0;
            }    

            world_value[index] = final_value;
            id_matrix[index] = final_id;
            return;
        } else{
        if (fighters < 1){
            return;
        }

        
        if(starting_value > 0){
            float tot_energy = (starting_value > enemy_energy) ? enemy_energy : starting_value;
            tot_energy = tot_energy * 4; 
            int window_size = 3;
            int radius = window_size / 2;

            int center_x = index % dim_world;
            int center_y = index / dim_world;

            int filter_x;
            int filter_y;
            int filter_index;
            float filter_energy = tot_energy /( (window_size * window_size)-1);

            for (int i = 0; i < window_size; i++){
                for (int j = 0; j < window_size; j++){
                    filter_x = (center_x - radius + i + dim_world) % dim_world;
                    filter_y = (center_y - radius + j + dim_world) % dim_world;

                    filter_index = filter_y * dim_world + filter_x;
                    if(filter_index != index){
                        atomicAdd(&world_value[filter_index], filter_energy);                            
                    }
                }
            } 
            atomicAdd(&world_value[index], - tot_energy);
            id_matrix[index] = 0;   

        }else{
            if(fighters == 1){
                final_value = max_value;
                final_id = max_id;
            }else{
                final_value = starting_value + max_value - ((enemy_energy - max_value) / (fighters - 1));
                final_id = max_id;                
            }
            world_value[index] = final_value;
            id_matrix[index] = final_id;

        }    
        world_signal[index] = 0;    
    }
}    
    
    //================================================================================

    // kernel that clean the obstacles and food around the creature
__global__ void clean_around_cells_kernel (float* world_value_d, int* world_id_d, int dim_world, int* cellule, int ncellule, int window_size){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= ncellule * window_size * window_size) return;

    int cell_idx = cellule[idx / (window_size * window_size)];

    int filter_idx = idx % (window_size * window_size);

    int center_x = cell_idx % dim_world;
    int center_y = cell_idx / dim_world;

    int filter_x = filter_idx % window_size;
    int filter_y = filter_idx / window_size;

    int radius =  window_size/2;

    int final_x = (center_x + (filter_x - radius) + dim_world) % dim_world;
    int final_y = (center_y + (filter_y - radius) + dim_world) % dim_world;


    int final_index = final_y * dim_world + final_x;

    if (final_index == cell_idx)return;

    world_value_d[final_index] = 0;
    world_id_d[final_index] = 0;

}    

 //============================================================================================
 
    //Wrapper add objects to world
void launch_add_objects_to_world(float* world_value_d, int* world_id_d, int dim_world,
                                int id, float min_value, float max_value, float threshold,
                                curandState curandStates[],
                                cudaStream_t stream) {

    dim3 blockDim(16, 16);
    dim3 gridDim((dim_world + 15) / 16, (dim_world + 15) / 16);

    add_objects_to_world_kernel<<<gridDim, blockDim, 0, stream>>>(
        world_value_d, world_id_d, dim_world,
        id, min_value, max_value, threshold,
        curandStates
    );

}

//Wrapper world update
void launch_world_update(
    float *world_value,
    int *id_matrix,
    float *world_signal,
    float *contribution_matrix, 
    int world_dim, 
    int number_of_creatures,
    float energy_decay,
    cudaStream_t stream
){


    int n_thread_per_block = 1024; 
    int thread_number = world_dim*world_dim;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;

    world_update_kernel<<<n_block,n_thread_per_block,0,stream>>>(
        world_value, 
        id_matrix,
        world_signal,
        contribution_matrix,
        world_dim,
        number_of_creatures,
        energy_decay
    );

}


// wrapper that launch clear space around cells
void launch_clean_around_cells(float* world_value_d, int* world_id_d, int dim_world, int* cellule, int* ncellule, int window_size, cudaStream_t stream){

    int n_thread_per_block = 1024; 
    int localNcellule = *ncellule;
    int thread_number = localNcellule * window_size * window_size;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;


    clean_around_cells_kernel<<<n_block,n_thread_per_block,0,stream>>>(world_value_d, world_id_d, dim_world, cellule, localNcellule, window_size);
}

// ===============================================================================================================================================================

__global__ void alive_cells_builder_kernel(int* mondo_id, int* alive_cells_d, int thread_number){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= thread_number)return;
    int ID = mondo_id[idx];
    if(ID > 0){
        alive_cells_d[idx] = idx;
    }else{
        alive_cells_d[idx] = -1;
    }

}

void compact_with_thrust(int* mondo_id, int* alive_cells_d, int dim_world, int &new_size, cudaStream_t stream) {

    int n_thread_per_block = 1024; 
    int thread_number = dim_world*dim_world;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;

    alive_cells_builder_kernel<<<n_block,n_thread_per_block,0,stream>>>(
        mondo_id,
        alive_cells_d,
        thread_number
    );

    // 1) Wrap del raw pointer in un device_ptr di Thrust
    thrust::device_ptr<int> dev_ptr(alive_cells_d);

    // 2) Lancio remove_if con predicato “x < 0” → sposterà tutti gli x >= 0 in testa
    //    e restituirà un iterator al “new end”.
    auto new_end = thrust::remove_if(
        dev_ptr, 
        dev_ptr + thread_number, 
        thrust::placeholders::_1 < 0
    );

    // 3) Calcolo la nuova dimensione
    new_size = static_cast<int>(new_end - dev_ptr);
    // Da questo momento gli elementi validi stanno in alive_cells_d[0..new_size-1].
}