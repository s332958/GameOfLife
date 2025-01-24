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

    //compute cell to modify with convolution  (one thread per cell)
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int cell_index = y*dim_world+x;
    int ID = id_matrix[cell_index];
    //stop thread out of bound
    if (y>=dim_world || x>=dim_world) return;

    // 0 obstacle contribution, 1 world contribution, 1> creature contribution
    int dim_points = number_of_creatures+WORLD_OBJECT+1;
    float points[MAX_CREATURES] = {};

    //compute limit of filter
    int lim = dim_filter/2;
    //float kernelsum = 0;

    for(int i = -lim; i<=lim; i++){
        for(int j = -lim; j<=lim; j++){

            //compute cell index of neighbors in toroidal world
            int world_y = ((i + y + dim_world) % dim_world);
            int world_x = ((j + x + dim_world) % dim_world);

            //get cell neighbors and filter cell on that cell
            int world_cell = world_y * dim_world + world_x;
            int filter_cell = (lim + i) * dim_filter + (lim + j);
            float value_contribution = filter[filter_cell] * world[world_cell]/255;
            //compute vector of contribution from creatures and obstacles for the cell
            int world_id_cell_contribution = id_matrix[world_cell] + WORLD_OBJECT; 
            points[world_id_cell_contribution] += value_contribution;            
        }
    }
    
    //compute max contribution from creatures, obstacles and world (world  and obstacles contributes are unused by default)
    //the greater contribution of creture give the cell id
    float final_point = 0;
    int final_id_cell = ID;
    int first_creature = WORLD_OBJECT+1;


    
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
      




    //activation function
    
    float m = 0.35, s = 0.08, T = 10;
    float growth_value = exp(-pow(((final_point - m) / s),2)/ 2 )*2-1;
    float increment = (1.0 / T) * growth_value;
    final_point = fmaxf(0.0, fminf(1.0, world[cell_index]/255 + increment)); 

    final_point = final_point*255;   

    if (final_point == 0){
        final_id_cell = 0;
    }

    /*
    if(final_id_cell != 0){
        printf("%.6f", increment);
    } 
    if(cell_index == 100){
        int size = sizeof(points) / sizeof(points[1]);
        for (int i = 0; i < size; i++) {
            printf("%.2f, ", points[i]);
        }
        printf(" | ");
        printf("     %f     ",world[cell_index]);
        printf("     %f     ",final_point);
        printf("     %f     ",kernelsum);
    }
    */
    //check obstacles is used for decide wich cell need to be modify (obstacles cell remain the same)
    //bool check_obstacle = !(bool)(1 + ID);
    //printf("| %f %d |", final_point ,final_id_cell );
    //generate new world_matrix and matrix_id
    world_out[cell_index] = (int)final_point;                      //*(!check_obstacle)+ world[cell_index]*(check_obstacle);
    id_matrix_out[cell_index] = final_id_cell;                  //*(!check_obstacle) + ID*(check_obstacle);

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
                                            int creature_id, int *number_of_creaure, cudaStream_t stream){
    
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    //computation number of thread and block for launch kernel (use max thread for dimension before launch new block)
    int n_thread, n_block;
    n_block = dim_world/properties.maxThreadsDim[0] +1;
    if(n_block==1) n_thread = dim_world;
    else n_thread = dim_world/n_block +1;

    dim3 thread_number = dim3(n_block,n_block);
    dim3 block_number = dim3(n_thread,n_thread);
    
    //launch kernel for adding creature to world
    add_creature_to_world<<<block_number,thread_number,0,stream>>>(creature,world,id_matrix,dim_creature,dim_world,pos_x,pos_y,creature_id);
    *number_of_creaure = *number_of_creaure+1;
    cudaStreamSynchronize(stream);
    printf("wrap add creature: %s\n",cudaGetErrorString(cudaGetLastError()));

}

//function for prepare and launch convolution
extern "C" void wrap_convolution(float *world, int *id_matrix, float* filter, float *world_out, int *id_matrix_out, 
                            int dim_world, int dim_filter, int number_of_creatures, cudaStream_t stream){

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    //computation number of thread and block for launch kernel (use max thread for dimension before launch new block)
    int n_thread, n_block;
    n_block = dim_world/properties.maxThreadsDim[0] +1;
    if(n_block==1) n_thread = dim_world;
    else n_thread = dim_world/n_block +1;

    dim3 thread_number = dim3(n_block,n_block);
    dim3 block_number = dim3(n_thread,n_thread);

     //launch kernel for adding creature to world
    convolution<<<block_number,thread_number,0,stream>>>(world,id_matrix,filter,world_out,id_matrix_out,dim_world,dim_filter,number_of_creatures);
    cudaStreamSynchronize(stream);
    printf("wrap convolution: %s\n",cudaGetErrorString(cudaGetLastError()));

}