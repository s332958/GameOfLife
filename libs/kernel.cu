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
__global__ void convolution(float *mondo_creature, float *world, int *id_matrix, float* filter,
                            int dim_world, int dim_filter, int number_of_creatures, int convolution_iter)
    {
        (mondo_creature, mondo, mondo_signal, cellule_cu, dim_mondo, number_of_creatures, n_thread, dim_output_mov, convolution_iter);

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= n_thread) return;

    int dim_output = dim_output_mov * dim_output_mov + 1;

    int cell_index = index / dim_output;
    int output_index = index - cell_index * dim_output;
    int ID = cellule_cu[cell_index].ID;
    
    float output = cellule_cu[cell_index].output[output_index];
    
    int centro_index = cellule_cu[cell_index].index;
    int centro_x = cellule_cu[cell_index].center_x;
    int centro_y = cellule_cu[cell_index].center_y;

    if (output_index = dim_output){
        mondo_signal[center_index] = output;
        return;
    }

    int filtro_x = output_index % dim_output;
    int filtro_y = output_index / dim_output;
    int radius = dim_output_mov / 2;

    int filtro_x_mondo = (filtro_x - radius + centro_x) % dim_mondo;
    int filtro_y_mondo = (filtro_y - radius + centro_y) % dim_mondo; 
    
    int filtro_index = filtro_y * dim_world + filtro_x;
    
    atomicAdd(&mondo_creature[(ID * dim_world * dim_world) + filtro_index], output * max_increments);       
    atomicAdd(&mondo_creature[(ID * dim_world * dim_world) + centro_index], -output * max_increments);
    
    /*
    //if( ID<1 ||filtro_x < 0 || filtro_y < 0 || centro_index < 0 || centro_index >= dim_world*dim_world || filtro_x >= dim_world || filtro_y >= dim_world) printf("ERRORE (%d, %d, %d)",filtro_x,filtro_y,centro_index);

    if(centro_index == filtro_index){
        atomicAdd(&world[centro_index], decay);
        //printf("DECADO \n");
        return;
    } 
    
    //if(world[centro_index] < max_increments*(dim_filter*dim_filter-1)) return;
    if(id_matrix[filtro_index] == ID && world[filtro_index] > 255.0f) return;
    
    int index = centro_y*dim_filter + centro_x*dim_filter + filtro_y*dim_filter * filtro_x;
    curandState state;
    curand_init(1, index, 0, &state);
    float random_number = curand_uniform(&state);
    //printf("%f %d,%d \n",random_number,blockIdx.x,blockDim.y);
    float increment = random_number*max_increments/100*world[centro_index];

    atomicAdd(&mondo_creature[(ID * dim_world * dim_world) + filtro_index], increment);
    atomicAdd(&world[centro_index], -increment);    
    */
    

}

__global__ void mondo_cu_update(float *mondo_creature, float *world, int *id_matrix, 
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


__global__ void zero_all(float *mondo_creature, int dim_world, int n_thread){
    index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index > n_thread) return;
    mondo_creature[index] = 0.0f;
}

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

//function for prepare and launch convolution
extern "C" void wrap_convolution(Cellula *cellule_cu, float *mondo_creature, float *mondo, int *id_matrix, int dim_world,
                                 int number_of_creatures, int cellCount, int dim_output, int convolution_iter, cudaStream_t stream);

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    int n_thread_per_block = properties.maxThreadsPerBlock;  
    int n_thread = dim_world * dim_world * number_of_creatures;
    int n_block = n_thread / n_thread_per_block;
    if(n_block%thread_per_dimension!=0 || n_block==0) n_block=n_block+1; 


    zero_all<<<n_block, n_thread_per_block,0,stream>>>(mondo_creature, dim_world, n_thread);
    
    
    cudaStreamSynchronize(stream);
    

    //printf("world: %p \nid_matrix: %p \nfilter: %p \nworld_save: %p \nid_matrix_save: %p \n",world,id_matrix,filter,world_save,id_matrix_save);
    //printf("blocchi: (%d,%d), thread: (%d,%d) \n",block_number.x,block_number.y,thread_number.x,thread_number.y);

    //launch kernel 
    //printf("mondo_creatura %p\n, world %p\n, id_creature %p\n, world_save: %p\n, id_matrix_save: %p\n, filter: %p \n",mondo_creature,world,id_matrix,world_save,id_matrix_save,filter);
    
    int n_thread =  cellCount * (dim_output*dim_output + 1);
    int n_block = n_thread / n_thread_per_block;
    if(n_block%thread_per_dimension!=0 || n_block==0) n_block=n_block+1; 
    
    convolution<<<n_block,n_thread_per_block,0,stream>>>(mondo_creature, mondo, cellule_cu, dim_world, number_of_creatures, n_thread, dim_output, convolution_iter);

    //cudaStreamSynchronize(stream);

    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("wrap convolution: %s\n",cudaGetErrorString(cudaGetLastError()));
    //cudaStreamSynchronize(stream);

    n_thread_per_block = properties.maxThreadsPerBlock;  
    int thread_per_dimension = sqrt(n_thread_per_block);
    n_block = dim_world / thread_per_dimension;
    if(n_block==0) n_block = 1;
    thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    block_number = dim3(n_block, n_block); 

    mondo_cu_update<<<block_number,thread_number,0,stream>>>(mondo_creature, world,id_matrix,filter,world_save,id_matrix_save,dim_world,number_of_creatures,convolution_iter);

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







