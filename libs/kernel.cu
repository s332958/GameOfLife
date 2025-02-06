#include <cuda_runtime.h>
#include <iostream>
#include "kernel.cuh"

#define MAX_CREATURES 64
//types of obstacles in the world (setup for 1 type)
#define WORLD_OBJECT 1

//function for convolution
__global__ void convolution(float *world, int *id_matrix, float* filter, float *world_out, int *id_matrix_out, 
                            int dim_world, int dim_filter, int number_of_creatures)
    {
    int radius_filter = (dim_filter-1)/2;
    int centro_x = blockIdx.x;
    int centro_y = blockIdx.y;
    int filtro_x = threadIdx.x - radius_filter;
    int filtro_y = threadIdx.y - radius_filter;

    int cell_index = centro_y * dim_world + centro_x;
    int filter_index = threadIdx.y * dim_filter + threadIdx.x;

    int ID = id_matrix[cell_index];
    bool libero = ID == 0;
    bool ostacolo = ID == - 1;
    
    if (ostacolo){return;}
    //stop thread out of bound
    //if (centro_x>=dim_world || centro_y>=dim_world || filtro_x + radius_filter>=dim_filter || filtro_y + radius_filter>=dim_filter) return;

    int dim_points = number_of_creatures + WORLD_OBJECT;

    __shared__ float points[MAX_CREATURES + WORLD_OBJECT];
    __shared__ float value_filter_normalizzation;
    
    value_filter_normalizzation = 0.0f;  

    if (dim_world*threadIdx.y + threadIdx.x <= dim_points){        
        points[dim_world*threadIdx.y + threadIdx.x] = 0.0f;    
    }

    int world_x = ((centro_x + filtro_x + dim_world) % dim_world);
    int world_y = ((centro_y + filtro_y + dim_world) % dim_world);
    int world_cell = world_y * dim_world + world_x;
    int id_filter = 0;
    __syncthreads();
    id_filter = ID*(!libero) + id_matrix[world_cell]*(libero) - 1;
    int filter_cell = id_filter*dim_filter*dim_filter + filter_index;

    float value_contribution = filter[filter_cell] * world[world_cell]/255;
    int world_id_cell_contribution = id_matrix[world_cell]; 
    float ooo = filter[filter_cell];

    atomicAdd(&value_filter_normalizzation,ooo);  
    atomicAdd(&points[world_id_cell_contribution],value_contribution); 


    __syncthreads();


    if(threadIdx.y * dim_filter + threadIdx.x == 0){

        float value = 0;
        int final_id_cell = ID;
        int first_creature = 1;
        
        float best_point = 0;
        float enemy = 0;
        int best_creature = 0;

        for(int i=first_creature;i<=dim_points;i++){
            if(points[i] - best_point > 0){
                best_creature = i;
                best_point = points[i];
            }
            /*
            bool differentID = ID != i;
            bool greater = points[i] - best_point > 0;

            best_point = greater * points[i] + !greater * best_point;
            best_creature = greater * i + !greater * ID;
            enemy += points[i] * differentID;
            */
        }
        if (ID == 0){
            final_id_cell = best_creature;
        }
        for(int i=first_creature;i<=dim_points;i++){
            if(final_id_cell != i){
                enemy += points[i];
            }
            else{
                value += points[i];
            }
        }
        value = value - enemy;
        //value = libero * best_point + !libero * (points[ ID ] - enemy);
        value = value/value_filter_normalizzation;        
        //final_id_cell = libero * best_creature + !libero * final_id_cell;


        

        //activation function        
        //float m = 0.135, s = 0.015, T = 10;
        float m = 0.5, s = 0.15, T = 10;
        float growth_value = exp(-pow(((value - m) / s),2)/ 2 )*2-1;
        float increment = (1.0 / T) * growth_value;
        float final_point = fmaxf(0.0, fminf(1.0, world[cell_index]/255 + increment)); 

        final_point = final_point*255;   
        bool alive = final_point>0;
        final_id_cell = alive * final_id_cell;
        /*
        if (final_id_cell == 0){
            final_point = 0;
        }
        */

        world_out[cell_index] = (int)final_point;                   
        id_matrix_out[cell_index] = final_id_cell;               

    }
    

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
    if(n_block%thread_per_dimension!=0) n_block=n_block+1; 

    dim3 thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    dim3 block_number = dim3(n_block, n_block); 
    
    //launch kernel
    add_creature_to_world<<<block_number,thread_number,0,stream>>>(creature,world,id_matrix,dim_creature,dim_world,pos_x,pos_y,creature_id);
    *number_of_creaure = *number_of_creaure+1;
    //cudaStreamSynchronize(stream);
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("wrap add creature: %s\n",cudaGetErrorString(cudaGetLastError()));

}

//function for prepare and launch convolution
extern "C" void wrap_convolution(float *world, int *id_matrix, float* filter, float *world_out, int *id_matrix_out, 
                                    int dim_world, int dim_filter, int number_of_creatures, cudaStream_t stream
                                ){

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    dim3 thread_number = dim3(dim_filter,dim_filter);
    dim3 block_number = dim3(dim_world,dim_world);

    //launch kernel 
    convolution<<<block_number,thread_number,0,stream>>>(world,id_matrix,filter,world_out,id_matrix_out,dim_world,dim_filter,number_of_creatures);
    //cudaStreamSynchronize(stream);
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("wrap convolution: %s\n",cudaGetErrorString(cudaGetLastError()));

}























__global__ void creature_evaluation(
    float *world, int *id_matrix,
    int *creature_occupations, float *creature_values,
    int number_of_creatures, int dim_world
){
    
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    int world_cell = tx + ty*dim_world;

    if(world_cell >= dim_world*dim_world) {
        return;
    }
    else{
        float val = world[world_cell];
        int id = id_matrix[world_cell];
        atomicAdd(&creature_occupations[id+WORLD_OBJECT],1);
        atomicAdd(&creature_values[id+WORLD_OBJECT],val);
        //printf("%d: occupation: %d totale %f \n",id,creature_occupations[id+WORLD_OBJECT],creature_values[id+WORLD_OBJECT]);
    }

}

extern "C" void wrap_creature_evaluation(float *world, int *id_matrix, 
                                        int *creature_occupations, float *creature_values, 
                                        int dim_world, int number_of_creatures, cudaStream_t stream
                                ){

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    int n_thread = 32;
    int n_block = dim_world / n_thread;
    if(n_block==0) n_block++;
    dim3 block = dim3(n_block,n_block);
    dim3 thread = dim3(n_thread,n_thread);

    //launch kernel 
    //creature_evaluation<<<block,thread,0,stream>>>(world,id_matrix,creature_occupations,creature_values,number_of_creatures,dim_world);
    //cudaStreamSynchronize(stream);
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("wrap creature evaluation: %s\n",cudaGetErrorString(cudaGetLastError()));

}









