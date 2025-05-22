#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include "kernel.cuh"
#include "libs/Cellula.cuh"

#define MAX_CREATURES 64
//types of obstacles in the world (setup for 1 type)
#define WORLD_OBJECT 1

#define max_increments 5.0f
#define decay -5.0f

//function for convolution


__global__ void add_base_food(float*mondo, int *id_matrix, float max_food, int dim_world){
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = cy * dim_world + cx;
    if (cx >= dim_world || cy >= dim_world) return;
    if (id_matrix[index] != 0) return;
    curandState state;
    curand_init(1, index, 0, &state);
    float random_number = curand_uniform(&state);
    if (random_number < 0.99999) return;
    float increment = random_number*max_food;
    mondo[index] = 255.0f;
    
}    
__global__ void cellule_cleanup(Cellula *cellule_cu, int *id_matrix, int* cellCount, int* mask_cu){
    __shared__ int dim_paral = 10;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx > cellCount) return;
    
    if(id_matrix[cellule_cu[idx].index] <= 0){
       cellule_cu[idx].alive = false; 
    }    

    mask_cu[idx] = cellule_cu[idx].alive;
    id_sort_x = idx % dim_paral;
    id_sort_y = idx / dim_paral;
    
    if (id_sort_x == 0){
        int increment = 0;
        for (int i = 0; i < dim_paral; i++){
            if(id_sort_y + i < cellCount){
                increment = increment + mask_cu[id_sort_y*dim_paral + i];
                mask_cu[id_sort_y*dim_paral + i] = increment;
            }                
        }    
    }    
    __syncthreads();

    if (id_sort_y != 0){
        mask_cu[id_sort_y*dim_paral + id_sort_x] = mask_cu[id_sort_y*dim_paral + id_sort_x] + mask_cu[id_sort_y*dim_paral - 1];
    }    
    __syncthreads();

    if (cellule[idx].alive){
        cellule[mask[idx] - 1] = cellule[idx];
    }    

    if (idx != 0) return;
    cellCount = mask_cu[cellCount - 1];
}    

    //function for add creture
__global__ void add_creature_to_world(float* creature, float *world, int *id_matrix, int dim_creature, int dim_world, int pos_x, int pos_y, int creature_id){

    //compute cell to modify with convolution  (one thread per cell)
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    //stop thread out of bound
    if (y>=dim_creature || x>=dim_creature) return;

    //compute cell index in toroidal world
    int world_x = (pos_x+x)%dim_world;
    int world_y = (pos_y+y)%dim_world;
    int world_cell = (world_y)*dim_world +(world_x);

    //check looking for empty cell 
    bool check_empty = !(bool)(id_matrix[ world_cell ]);

    //update only empty cell (if they are already ocupated ignore them)
    world[ world_cell ] += creature[ y*dim_creature + x ] * (float)check_empty;
    bool alive = world[ world_cell ];
    id_matrix[ world_cell ] = alive * creature_id * (float)check_empty + id_matrix[ world_cell ] * (float)!check_empty;

}    

//function for prepare and launch add creture
extern "C" void wrap_add_creature_to_world(float* creature, float *world, int *id_matrix, 
                                            int dim_creature, int dim_world, int pos_x, int pos_y, 
                                            int creature_id, int *number_of_creaure, cudaStream_t stream
                                            ){
    
    cudaDeviceProp properties;                                            
    cudaGetDeviceProperties(&properties,0);

    //computation number of thread and block for launch kernel (use max thread for dimension before launch new block)
    int n_thread_per_block = properties.maxThreadsPerBlock;  
    int thread_per_dimension = sqrt(n_thread_per_block);
    int n_block = dim_world / thread_per_dimension;
    if(n_block%thread_per_dimension!=0 || n_block==0) n_block=n_block+1; 

    dim3 thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    dim3 block_number = dim3(n_block, n_block); 

    //std::cout << "thread e blocchi per add creature" << thread_per_dimension << "   " << n_block << "\n";
    
    //launch kernel
    add_creature_to_world<<<block_number,thread_number,0,stream>>>(creature,world,id_matrix,dim_creature,dim_world,pos_x,pos_y,creature_id);
    *number_of_creaure = *number_of_creaure+1;
    //cudaStreamSynchronize(stream);
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("wrap add creature: %s\n",cudaGetErrorString(cudaGetLastError()));

}  
extern "C" void wrap_add_base_food(float *world, int *id_matrix, float max_food, int dim_world){
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    int n_thread_per_block = properties.maxThreadsPerBlock;  
    int thread_per_dimension = sqrt(n_thread_per_block);
    int n_block = dim_world / thread_per_dimension;
    if(n_block==0) n_block++;
    dim3 block = dim3(n_block,n_block);
    dim3 thread = dim3(thread_per_dimension,thread_per_dimension);

    add_base_food<<<block,thread>>>(world,id_matrix,max_food,dim_world);

}

__global__ void creature_evaluation(
    float *world, int *id_matrix,
    int *creature_occupations, float *creature_values,
    int n_creature_obstacles, int dim_world)
    {  

    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
    int world_cell = tx + ty*dim_world;

    __shared__ float score_blocks[MAX_CREATURES];
    __shared__ float volume_blocks[MAX_CREATURES];
    
    //azzero i punteggi nella shared memory
    if (threadIdx.y == 0 && threadIdx.x < n_creature_obstacles){        
        score_blocks[threadIdx.x] = 0.0f; 
        volume_blocks[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    //controllo di non eccedere fuori dai limiti e se sono ostacoli o celle vuote non le considero
    if (tx < dim_world && ty < dim_world) {
        float val = world[world_cell];
        int id = id_matrix[world_cell];
        if (id >= 0 && id < n_creature_obstacles) { 
            atomicAdd(&score_blocks[id], val);
            atomicAdd(&volume_blocks[id], 1);
        }
    }

    __syncthreads();

    //sommo i valori nelle memorie condivise e li metto nella globale
    if(threadIdx.x == 0 && threadIdx.y == 0){
        for(int i = 0; i < n_creature_obstacles; i++){
            atomicAdd(&creature_occupations[i],volume_blocks[i]);
            atomicAdd(&creature_values[i],score_blocks[i]);
    
        }
    }

}

extern "C" void wrap_creature_evaluation(float *world, int *id_matrix, 
                                        int *creature_occupations, float *creature_values, 
                                        int dim_world, int n_creature_obstacles, cudaStream_t stream
                                ){

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    int n_thread = 32;
    int n_block = dim_world / n_thread;
    if(n_block==0) n_block++;
    dim3 block = dim3(n_block,n_block);
    dim3 thread = dim3(n_thread,n_thread);

    //printf("blocchi: (%d,%d), thread: (%d,%d) \n",block.x,block.y,thread.x,thread.y);

    //launch kernel 
    creature_evaluation<<<block,thread,0,stream>>>(world,id_matrix,creature_occupations,creature_values,n_creature_obstacles,dim_world);
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("errori post creature_evalution: %s\n",cudaGetErrorString(cudaGetLastError()));
    cudaStreamSynchronize(stream);
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("wrap creature evaluation: %s\n",cudaGetErrorString(cudaGetLastError()));

}

extern "C" int wrap_cellule_cleanup(Cellula *cellule_cu, int* cellCount, int* mask_cu){
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    int n_thread_per_block = properties.maxThreadsPerBlock;  
    int thread_per_dimension = sqrt(n_thread_per_block);
    int n_block = sqrt(cellCount) / thread_per_dimension;
    if(n_block==0) n_block++;
    dim3 block = dim3(n_block,n_block);
    dim3 thread = dim3(thread_per_dimension,thread_per_dimension);

    add_base_food<<<block,thread>>>(cellule_cu, cellCount, mask_cu);

}







_global__ void kernel_mondo_cu_update(float *mondo_creature, float *world, int *id_matrix, 
                                int dim_world, int number_of_creatures, int *cellCount, Cellula *cellule_cu, int convolution_iter)
    {                        
    int center_x = blockIdx.x*blockDim.x + threadIdx.x;    
    int center_y = blockIdx.y*blockDim.y + threadIdx.y;
    int index = center_y * dim_world + center_x;
    
    if(index >= dim_world*dim_world) return;
    
    
    int ID = id_matrix[index];
    float starting_value = world[index];
    
    if(ID == -1) return;

    float final_value = starting_value;
    int final_id = ID;

    int max_id = 0;
    int max_value = 0;
    float ally_energy = 0;
    float enemy_energy = 0;
    for(int i = 1; i <= number_of_creatures; i++){
        int value = mondo_creature[i * dim_world * dim_world + index]; 
        if (i == ID){
            ally_energy = value;      
        }    
        else{
            enemy_energy += value;
        }                      
        if (value > max_value){
            max_value = value;
            max_id = i;
        }    
    }    
    if (ID == 0){
        if (enemy_energy > 0){
            final_value = starting_value * (max_value / enemy_energy) + max_value;
            final_id = max_id;

            pos = atomicAdd(cellCount, 1);
            Cellula newCell;
            nuova.index = index;
            nuova.ID = final_id;
            nuova.alive = true;
            nuova.dim_visione = 9; //dim_visione
            nuova.center_x = index % dim_mondo;
            nuova.center_y = index / dim_mondo;

            cellule_cu[pos] = newCell;
        }    
    }    
    else{
        if (starting_value + ally_energy - enemy_energy < 0){
            final_value = abs(starting_value + ally_energy - enemy_energy);
            final_id = 0;
        }            
        else{
            final_value = starting_value + ally_energy - enemy_energy;
        }    
    }    

    /*
    final_value = fmaxf(0.0, fminf(255.0, final_value)); 
    int index_save = convolution_iter*dim_world*dim_world + index;
    world_save[index_save] = (unsigned char)final_value;
    id_matrix_save[index_save] = (unsigned char)final_id;
    */ 
   if(final_value < 20.0f){
        final_id = 0;
   }     

   if(final_value > 255.0f){
        final_value = 255.0f;
    }    

    world[index] = final_value;                   
    id_matrix[index] = (int)final_id; 


}    

//function for prepare and launch convolution
extern "C" void wrap_mondo_cu_update(float *mondo_creature, float *world, int *id_matrix, int dim_world, int number_of_creatures,
                                     int *cellCount, Cellula *cellule_cu, int convolution_iter);{


    n_thread_per_block = properties.maxThreadsPerBlock;  
    int thread_per_dimension = sqrt(n_thread_per_block);
    n_block = dim_world / thread_per_dimension;
    if(n_block==0) n_block = 1;
    thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    block_number = dim3(n_block, n_block); 

    kernel_mondo_cu_update<<<block_number,thread_number,0,stream>>>(mondo_creature, world,id_matrix,filter,world_save,id_matrix_save,dim_world,number_of_creatures,convolution_iter);

}

_
__global__ void kernel_zero_mondocreature(float *mondo_creature, int dim_world, int n_thread){
    index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index > n_thread) return;
    mondo_creature[index] = 0.0f;
}    
extern "C" void wrap_zero_mondocreature(float* mondo_creature, int* dim_world){
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    int n_thread_per_block = properties.maxThreadsPerBlock;  
    int n_thread = dim_world * dim_world * number_of_creatures;
    int n_block = n_thread / n_thread_per_block;
    if(n_block%thread_per_dimension!=0 || n_block==0) n_block=n_block+1; 


    kernel_zero_mondocreature<<<n_block, n_thread_per_block,0,stream>>>(mondo_creature, dim_world, n_thread);
    
    

}    


__global__ void cellule_cleanup(Cellula *cellule_cu, int *id_matrix, int* cellCount, int* mask_cu){
    __shared__ int dim_paral = 10;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx > cellCount) return;
    
    if(id_matrix[cellule_cu[idx].index] <= 0){
       cellule_cu[idx].alive = false; 
    }    

    mask_cu[idx] = cellule_cu[idx].alive;
    id_sort_x = idx % dim_paral;
    id_sort_y = idx / dim_paral;
    
    if (id_sort_x == 0){
        int increment = 0;
        for (int i = 0; i < dim_paral; i++){
            if(id_sort_y + i < cellCount){
                increment = increment + mask_cu[id_sort_y*dim_paral + i];
                mask_cu[id_sort_y*dim_paral + i] = increment;
            }                
        }    
    }    
    __syncthreads();

    if (id_sort_y != 0){
        mask_cu[id_sort_y*dim_paral + id_sort_x] = mask_cu[id_sort_y*dim_paral + id_sort_x] + mask_cu[id_sort_y*dim_paral - 1];
    }    
    __syncthreads();

    if (cellule[idx].alive){
        cellule[mask[idx] - 1] = cellule[idx];
    }    

    if (idx != 0) return;
    cellCount = mask_cu[cellCount - 1];
} 

extern "C" int wrap_cellule_cleanup(Cellula *cellule_cu, int* cellCount, int* mask_cu){
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    int n_thread_per_block = properties.maxThreadsPerBlock;  
    int thread_per_dimension = sqrt(n_thread_per_block);
    int n_block = sqrt(cellCount) / thread_per_dimension;
    if(n_block==0) n_block++;
    dim3 block = dim3(n_block,n_block);
    dim3 thread = dim3(thread_per_dimension,thread_per_dimension);

    add_base_food<<<block,thread>>>(cellule_cu, cellCount, mask_cu);

}