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
    int cell_index = centro_y*dim_world+centro_x;

    int ID = id_matrix[cell_index];
    
    //stop thread out of bound
    if (centro_x>=dim_world || centro_y>=dim_world || filtro_x + radius_filter>=dim_filter || filtro_y + radius_filter>=dim_filter) return;

    int dim_points = number_of_creatures+WORLD_OBJECT+1;

    __shared__ float points[MAX_CREATURES];

    if (dim_world*threadIdx.y + threadIdx.x < dim_points){        
        points[dim_world*threadIdx.y + threadIdx.x] = 0.0f;         
    }


    __syncthreads();

    int world_x = ((centro_x + filtro_x + dim_world) % dim_world);
    int world_y = ((centro_y + filtro_y + dim_world) % dim_world);

    int world_cell = world_y * dim_world + world_x;
    int filter_cell = (radius_filter + filtro_y) * dim_filter + (radius_filter + filtro_x);
    float value_contribution = filter[filter_cell] * world[world_cell]/255;

    int world_id_cell_contribution = id_matrix[world_cell] + WORLD_OBJECT; 

    atomicAdd(&points[world_id_cell_contribution],value_contribution);            

    __syncthreads();


    if(threadIdx.x + dim_world*threadIdx.y == 0){

        float final_point = 0;
        int final_id_cell = ID;
        int first_creature = WORLD_OBJECT+1;

        float best_point = 0;
        float enemy = 0;
        int best_creature = 0;
        bool libero = !ID;

        for(int i=first_creature;i<dim_points;i++){
            bool greater = points[i] - best_point > 0;
            bool differentID = ID != i;

            best_point = greater * points[i] + !greater * best_point;
            best_creature = greater * i + !greater * ID;
            enemy += points[i + WORLD_OBJECT] * differentID;
        }

        final_point = libero * best_point + !libero * (points[ ID+WORLD_OBJECT ] - enemy);
        final_id_cell = libero * (best_creature-WORLD_OBJECT) + !libero * final_id_cell;

        /*
        if(ID==0){
            for(int i=first_creature;i<dim_points;i++){
                if(final_point<points[i]){
                    final_point = points[i];
                    final_id_cell = i-WORLD_OBJECT;
                }
            }
        }
        else{
            float enemy = 0;
            for(int i=first_creature;i<dim_points;i++){
                if(ID != i){
                    enemy += points[i + WORLD_OBJECT];
                }
            }
            final_point = points[ID + WORLD_OBJECT] - enemy;
        }

        */

        //activation function        
        //float m = 0.135, s = 0.015, T = 10;
        float m = 0.135, s = 0.1, T = 100;
        float growth_value = exp(-pow(((final_point - m) / s),2)/ 2 )*2-1;
        float increment = (1.0 / T) * growth_value;
        final_point = fmaxf(0.0, fminf(1.0, world[cell_index]/255 + increment)); 

        final_point = final_point*255;   
        
        /*
        if (final_point == 0){
            final_id_cell = 0;
        }
        */

        bool alive = final_point > 0.0001;
        final_id_cell = alive * final_id_cell;

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

    //check looking for empty cell 
    bool check_empty = !(bool)(id_matrix[ (world_y)*dim_world +(world_x) ]);

    //update only empty cell (if they are already ocupated ignore them)
    world[ (world_y)*dim_world +(world_x) ] += creature[ y*dim_creature + x ] * (float)check_empty;
    id_matrix[ (world_y)*dim_world +(world_x) ] = creature_id * (float)check_empty + id_matrix[ (world_y)*dim_world +(world_x) ] * (float)!check_empty;

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
    
    //launch kernel for adding creature to world
    add_creature_to_world<<<block_number,thread_number,0,stream>>>(creature,world,id_matrix,dim_creature,dim_world,pos_x,pos_y,creature_id);
    *number_of_creaure = *number_of_creaure+1;
    cudaStreamSynchronize(stream);
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

     //launch kernel for adding creature to world
    convolution<<<block_number,thread_number,0,stream>>>(world,id_matrix,filter,world_out,id_matrix_out,dim_world,dim_filter,number_of_creatures);
    cudaStreamSynchronize(stream);
    if(cudaGetLastError()!=cudaError::cudaSuccess) printf("wrap convolution: %s\n",cudaGetErrorString(cudaGetLastError()));

}