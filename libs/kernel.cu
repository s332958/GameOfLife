#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include "kernel.cuh"

#define MAX_CREATURES 64
//types of obstacles in the world (setup for 1 type)
#define WORLD_OBJECT 1

#define max_increments 5.0f
#define decay -1.0f

//function for convolution
__global__ void convolution(float *mondo_creature, float *world, int *id_matrix, float* filter, unsigned char *world_save, unsigned char *id_matrix_save, 
                            int dim_world, int dim_filter, int number_of_creatures, int convolution_iter)
    {
    
    int radius_filter = dim_filter/2;
    int centro_x = blockIdx.x;
    int centro_y = blockIdx.y;
    int centro_index = centro_y * dim_world + centro_x;

    
    int ID = id_matrix[centro_index];
    if (ID < 1) return;
    
    
    int filtro_x = (threadIdx.x - radius_filter + centro_x) % dim_world;
    int filtro_y = (threadIdx.y - radius_filter + centro_y) % dim_world;
    int filtro_index = filtro_y * dim_world + filtro_x;

    if(centro_index == filtro_index){
        atomicAdd(&world[centro_index], decay);
        return;
    } 
    
    //if(world[centro_index] < max_increments*(dim_filter*dim_filter-1)) return;
    if(id_matrix[filtro_index] == ID && world[filtro_index] > 255.0f) return;
    
    int index = centro_y*dim_filter + centro_x*dim_filter + filtro_y*dim_filter * filtro_x;
    curandState state;
    curand_init(1, index, 0, &state);
    float random_number = curand_uniform(&state);
    float increment = random_number*max_increments/100*world[centro_index];

    atomicAdd(&mondo_creature[ID * dim_world * dim_world + filtro_index], increment);
    atomicAdd(&world[centro_index], -increment);
    

}

__global__ void mondo_cu_update(float *mondo_creature, float *world, int *id_matrix, float* filter, unsigned char *world_save, unsigned char *id_matrix_save, 
    int dim_world, int dim_block, int number_of_creatures, int convolution_iter)
    {
    int center_x = blockIdx.x*dim_block + threadIdx.x;
    int center_y = blockIdx.y*dim_block + threadIdx.y;
    int index = center_y * dim_world + center_x;
    
    if(index > dim_world*dim_world) return;
    
    
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
    world[index] = final_value;                   
    id_matrix[index] = (int)final_id;  

    }

__global__ void zero_all(float *mondo_creature, int dim_world, int thread_per_dimension, int number_of_creatures){
    int index = blockIdx.x * thread_per_dimension * thread_per_dimension + threadIdx.y * thread_per_dimension + threadIdx.x;
    if (index > dim_world * dim_world * (number_of_creatures+1)) return;
    mondo_creature[index] = 0.0f;
}

__global__ void add_base_food(float*mondo, int *id_matrix, float max_food, int thread_per_dimension, int dim_world){
    int cx = blockIdx.x * thread_per_dimension + threadIdx.x;
    int cy = blockIdx.y * thread_per_dimension + threadIdx.y;
    int index = cy * dim_world + cx;
    if (cx > dim_world || cy > dim_world) return;
    if (id_matrix[index] != 0) return;
    curandState state;
    curand_init(1, index, 0, &state);
    float random_number = curand_uniform(&state);
    if (random_number < 0.995) return;
    float increment = random_number*max_food;
    mondo[index] = 25500.0f;
    
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
extern "C" void wrap_convolution(float *mondo_creature, float *world, int *id_matrix, float* filter, unsigned char *world_save, unsigned char *id_matrix_save, 
                                    int dim_world, int dim_filter, int number_of_creatures, int convolution_iter, cudaStream_t stream
                                ){

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);    
    
    int n_thread_per_block = properties.maxThreadsPerBlock;  
    int thread_per_dimension = sqrt(n_thread_per_block);
    int n_block = (dim_world * dim_world * (number_of_creatures+1) + n_thread_per_block - 1) / n_thread_per_block;
    dim3 thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    dim3 block_number = dim3(n_block, 1); 

    zero_all<<<block_number, thread_number>>>(mondo_creature, dim_world, thread_per_dimension, number_of_creatures);
    
    
    
    
    int dim_filter_5 = 5;
    thread_number = dim3(dim_filter_5,dim_filter_5);
    block_number = dim3(dim_world,dim_world);

    //printf("world: %p \nid_matrix: %p \nfilter: %p \nworld_save: %p \nid_matrix_save: %p \n",world,id_matrix,filter,world_save,id_matrix_save);
    //printf("blocchi: (%d,%d), thread: (%d,%d) \n",block_number.x,block_number.y,thread_number.x,thread_number.y);

    //launch kernel 
    convolution<<<block_number,thread_number,0,stream>>>(mondo_creature, world,id_matrix,filter,world_save,id_matrix_save,dim_world,dim_filter_5,number_of_creatures,convolution_iter);

    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("wrap convolution: %s\n",cudaGetErrorString(cudaGetLastError()));
    //cudaStreamSynchronize(stream);

    n_thread_per_block = properties.maxThreadsPerBlock;  
    thread_per_dimension = sqrt(n_thread_per_block);
    n_block = dim_world / thread_per_dimension;
    thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    block_number = dim3(n_block, n_block); 

    mondo_cu_update<<<block_number,thread_number,0,stream>>>(mondo_creature, world,id_matrix,filter,world_save,id_matrix_save,dim_world,thread_per_dimension,number_of_creatures,convolution_iter);

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

    add_base_food<<<block,thread>>>(world,id_matrix,max_food,thread_per_dimension,dim_world);

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

    int n_thread = dim_world;
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









