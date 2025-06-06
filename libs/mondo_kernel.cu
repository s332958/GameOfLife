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
                                    int id, float min_value, float max_value, float threashold
                                ){
    
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int idx = x + y*dim_world;

    if(idx<dim_world*dim_world){

        if(world_id[idx]==0){
            // instantiate a random generator
            curandState state;
            curand_init(clock64(),threadIdx.x,0,&state);
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
            world_value[index] = -1;
            return;
        }
        
        // set the final cell value and id with the value of past step
        float final_value = starting_value;
        int final_id = ID;
        
        // use a for cicle to find the model that give more contribution
        int max_id = 0;
        float max_value = 0;
        float ally_energy = 0;
        float enemy_energy = 0;
        
        for(int i = 0; i < number_of_creatures; i++){
            // get the contribution value given id (i) and offset cell
            float value = contribution_matrix[i * dim_world * dim_world + index]; 
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

        // if the cell is empty
        if (ID == 0){
            if (enemy_energy > 0){
                
                // the final value is compute by the formula below
                // assing the id at the creature with most contribution
                final_value = starting_value * (max_value / enemy_energy) + max_value;
                final_id = max_id;

            }else{
                world_signal[index] = 0;
            }    
        } 
        
        // if the cell is occupied
        else{
            
            if (starting_value + ally_energy - enemy_energy < 0){
                // if the enemy cell is greater than staring value plus ally energy, set the final energy as the difference between the two value and final id equal 0 
                final_value = abs(starting_value + ally_energy - enemy_energy);
                final_id = 0;
            }            
            else{
                //if starting value plus ally cell is grater than enemy energy, set final vlaue as difference between the two value
                final_value = starting_value + ally_energy - enemy_energy;
            }

            // after the computation apply the energy decay
            final_value = final_value - energy_decay;    
        }    
        
        // if the final energy is less then 0.02 the id return to 0
        if(final_value < 0.02f){
           final_id = 0;
        }     
        
        // if the final value exceed the max (1.0) cap the value to max
        if(final_value > 1.0f){
            final_value = 1.0f;
        }    

        // assing to the cell teh final value
        world_value[index] = final_value;                   
        id_matrix[index] = final_id; 

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
                                cudaStream_t stream) {

    dim3 blockDim(16, 16);
    dim3 gridDim((dim_world + 15) / 16, (dim_world + 15) / 16);

    add_objects_to_world_kernel<<<gridDim, blockDim, 0, stream>>>(
        world_value_d, world_id_d, dim_world,
        id, min_value, max_value, threshold
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

// kernel that read the world and then save the index of the cell alive in cell alive vector
// after that start a reduction to compute the number of cells
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

    // write the value in the shared memory (0 if exceed the limits, that is for don't doing illigal memory access)
    shared_mem[tid] = (idx < world_dim_tot) ? is_alive : 0;
    __syncthreads();

    // Start the reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s+1 && tid+s+1 <blockDim.x) {
            shared_mem[tid] += shared_mem[tid + s+1];
            shared_mem[tid + s+1]=0;
        }
        __syncthreads();
    }

    // save the value of each reduction (one per block) in n_cell_alive
    if (tid == 0) {
        shared_mem[0] += shared_mem[1];
        shared_mem[1] = 0;
        atomicAdd(n_cell_alive, shared_mem[0]);
    }
}

// kernel that give the first part of compatting the array alive cell
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

    // load in shared
    if (global_idx < world_dim) {
        shared_mem[local_idx] = alive_cell_vector[global_idx];
        alive_cell_vector[global_idx] = 0;
    }

    __syncthreads();

    // thread 0 count the number of alive cell in his block and save it contigous cell in shared memory
    if (local_idx == 0) {
        int count = 0;
        for (int i = 0; i < blockDim.x; i++) {
            int val = shared_mem[i];
            if (val > 0) {
                shared_mem[blockDim.x + count] = val;
                count++;
            }
        }

        // save in the support vector the number of alive cell in his block
        shared_mem[2 * blockDim.x] = count;
        support_vector[blockIdx.x] = count;
    }

    __syncthreads();

    // Rewrite the compact value in global memory 
    int count = shared_mem[2 * blockDim.x];
    if (local_idx < count) {
        alive_cell_vector[global_idx] = shared_mem[blockDim.x + local_idx];
    }

}

// kernel that start the second part of compatting the alive cell vector
// this kernel MUST be launch with 1 block
__global__ void compact_cell_alive_kernel_pt2(int *alive_cell_vector, int *support_vector, int *n_alive_cell, int n_block, int dim_block){

    __shared__ int shared_mem[2];       
    //mem[0] starting index, mem[1] number of element 

    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    // the first cell in shared memory 0 rappresent the offset of the alive cell vector (where to start to write values) 
    // the second cell in shared memory is the number of alive cell in block
    // after read the value in the support vector, than update the value of the corrisponding block in the support vector
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

    // using threads for writing the alive cell value in adiacent cells, starting form the offset loaded before
    int idx_alive_cell_read = n_block*dim_block+idx;
    if(idx<shared_mem[1]){
        int offset = shared_mem[0];
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
    int *support_vector_d,
    cudaStream_t stream
) {
    int n_thread = 1024;
    if(world_dim_tot<n_thread) n_thread = world_dim_tot;
    int n_block = (world_dim_tot+n_thread-1) / n_thread;


    cudaMemsetAsync(n_cell_alive_d, 0, sizeof(int),stream);

    find_index_cell_alive_kernel<<<n_block,n_thread,sizeof(int)*(n_thread+1),stream>>>(
        world_id,
        alive_cell_vector,
        world_dim_tot,
        n_cell_alive_d
    );

    n_thread = 512;
    n_block = (world_dim_tot+n_thread-1) / n_thread;

    compact_cell_alive_kernel_pt1<<<n_block,n_thread,sizeof(int)*(n_thread*2+1),stream>>>(
        alive_cell_vector,
        support_vector_d,
        n_cell_alive_d,
        world_dim_tot
    );

    int block_dim_pt1 = n_thread;
    int n_block_pt1 = n_block;

    // this part is sequential, it is suppose to start to reorder the cell alive vector staring from index 0 to the n_alive_cell index
    // launch the kernel with only one block
    for(int i=0; i<n_block_pt1; i++){
        compact_cell_alive_kernel_pt2<<<1,n_thread,0,stream>>>(
            alive_cell_vector,
            support_vector_d,
            n_cell_alive_d,
            i,
            block_dim_pt1
        );
    }


}


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

void compact_with_thrust(int* mondo_id, int* alive_cells_d, int dim_world, int &new_size) {
    cudaMemset(alive_cells_d, 0, dim_world * dim_world * sizeof(float));

    int n_thread_per_block = 1024; 
    int thread_number = dim_world*dim_world;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;

    alive_cells_builder_kernel<<<n_block,n_thread_per_block,0>>>(
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