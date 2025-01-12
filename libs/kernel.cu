#include <cuda_runtime.h>
#include <iostream>
#include "kernel.cuh"

#define MAX_CREATURES 1024
#define WORLD_OBJECT 1

__global__ void convolution(float *world, int *id_matrix, float* filter, float *world_out, int *id_matrix_out, 
                            int dim_world, int dim_filter, int number_of_creatures)
    {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y>=dim_world || x>=dim_world) return;

    // 0 obstacle contribution, 1 world contribution, 1> creature contribution
    int dim_points = number_of_creatures+WORLD_OBJECT+1;
    float points[MAX_CREATURES] = {};

    int lim = dim_filter/2;

    for(int i = -lim; i<=lim; i++){
        for(int j = -lim; j<=lim; j++){

            int world_y = ((i + y + dim_world) % dim_world);
            int world_x = ((j + x + dim_world) % dim_world);

            int world_cell = world_y * dim_world + world_x;
            int filter_cell = (lim + i) * dim_filter + (lim + j);

            int world_id_cell_contribution = id_matrix[world_cell] + WORLD_OBJECT;
            float value_contribution = filter[filter_cell] * world[world_cell];
            points[world_id_cell_contribution] += value_contribution;

        }
    }

    //compute max contribution from creatures and contribution from obstacles and world (the last 2 contributes are unused by default)
    float final_point=0;
    int final_id_cell=0;
    int first_creature = WORLD_OBJECT+1;
    for(int i=first_creature;i<dim_points;i++){
        if(final_point<points[i]){
            final_point = points[i];
            final_id_cell = i-WORLD_OBJECT;
        }
    }
    final_point += (points[0]*0 + points[1]*0);

    int cell_index = y*dim_world+x;
    bool check_obstacle = !(bool)(1 + id_matrix[cell_index]);

    //generate new world_matrix and matrix_id
    world_out[cell_index] = final_point*(!check_obstacle) + world[cell_index]*(check_obstacle);
    id_matrix_out[cell_index] = final_id_cell*(!check_obstacle) + id_matrix[cell_index]*(check_obstacle);

}

__global__ void add_creature_to_world(float* creature, float *world, int *id_matrix, int dim_creature, int dim_world, int pos_x, int pos_y, int creature_id){

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    //printf("Start (%d,%d): (dim: %d)\n",x,y,dim_creature);
    if (y>=dim_creature || x>=dim_creature) return;

    int world_x = (pos_x+x)%dim_world;
    int world_y = (pos_y+y)%dim_world;

    bool check_empty = !(bool)(id_matrix[ (world_y)*dim_world +(world_x) ]);
    //printf("(%d,%d):%f \n",world_x,world_y,(float)check_empty);

    world[ (world_y)*dim_world +(world_x) ] += creature[ y*dim_creature + x ] * (float)check_empty;
    id_matrix[ (world_y)*dim_world +(world_x) ] = creature_id * (float)check_empty + id_matrix[ (world_y)*dim_world +(world_x) ] * (float)!check_empty;

}

extern "C" void wrap_add_creature_to_world(float* creature, float *world, int *id_matrix, int dim_creature, int dim_world, int pos_x, int pos_y, int creature_id){
    
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    int n_thread, n_block;
    n_block = dim_creature/properties.maxThreadsDim[0] +1;
    if(n_block==1) n_thread = dim_creature;
    else n_thread = dim_creature/n_block +1;

    dim3 thread_number = dim3(n_block,n_block);
    dim3 block_number = dim3(n_thread,n_thread);

    add_creature_to_world<<<block_number,thread_number>>>(creature,world,id_matrix,dim_creature,dim_world,pos_x,pos_y,creature_id);
    cudaDeviceSynchronize();

}

extern "C" void wrap_convolution(float *world, int *id_matrix, float* filter, float *world_out, int *id_matrix_out, 
                            int dim_world, int dim_filter, int number_of_creatures){

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);
    
    int n_thread, n_block;
    n_block = dim_world/properties.maxThreadsDim[0] +1;
    if(n_block==1) n_thread = dim_world;
    else n_thread = dim_world/n_block +1;

    dim3 thread_number = dim3(n_block,n_block);
    dim3 block_number = dim3(n_thread,n_thread);

    convolution<<<block_number,thread_number>>>(world,id_matrix,filter,world_out,id_matrix_out,dim_world,dim_filter,number_of_creatures);
    cudaDeviceSynchronize();

}