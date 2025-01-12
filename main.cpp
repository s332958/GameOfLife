#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "libs/kernel.cuh"
#include "libs/loader.h"
#include <iostream>
#include <iomanip>

int main(){
    printf("Start Main (Single Simulation): \n");

    //0: cpu init variable
    int dim_world, dim_filter, dim_creature, number_of_creatures=0;
    int *id_matrix;
    float *world, *filter, *creature;

    //0: load from file: world, id_matrix, filter and creature
    readWorld("data/world.txt",&dim_world,&world,&id_matrix);
    readMatrix("data/creature.txt",&dim_creature,&creature);
    readMatrix("data/filter.txt",&dim_filter,&filter);

    printing_matrix("creatura",creature,dim_creature);

    //0: cuda memory allocation
    float *world_cu, *filter_cu, *creature_cu;
    int *id_matrix_cu;

    cudaMalloc( (void**)&world_cu, dim_world*dim_world*sizeof(float) );
    cudaMalloc( (void**)&id_matrix_cu, dim_world*dim_world*sizeof(int) );
    cudaMalloc( (void**)&filter_cu, dim_filter*dim_filter*sizeof(float) );
    cudaMalloc( (void**)&creature_cu, dim_creature*dim_creature*sizeof(float) );

    cudaMemcpy( world_cu, world, dim_world*dim_world*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( id_matrix_cu, id_matrix, dim_world*dim_world*sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( filter_cu, filter, dim_filter*dim_filter*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( creature_cu, creature, dim_creature*dim_creature*sizeof(float), cudaMemcpyHostToDevice );

    //1: add creatures to world 
    int pos_x=0, pos_y=0, id_creature=1;
    wrap_add_creature_to_world( creature_cu, world_cu, id_matrix_cu, dim_creature, dim_world, pos_x, pos_y, id_creature);
    number_of_creatures++;

    //1: return world result for printing
    cudaMemcpy( world, world_cu, dim_world*dim_world*sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( id_matrix, id_matrix_cu, dim_world*dim_world*sizeof(int), cudaMemcpyDeviceToHost );

    //1: printing
    printing_world("world post add creature",world,id_matrix,dim_world);

    //2: start convolution 
    int *id_matrix_cu_out;
    float *world_cu_out;

    //2: allocation for convolution resault
    cudaMalloc( (void**)&world_cu_out, dim_world*dim_world*sizeof(float) );
    cudaMalloc( (void**)&id_matrix_cu_out, dim_world*dim_world*sizeof(int) );

    //2: run convolution
    wrap_convolution(world_cu, id_matrix_cu, filter_cu, world_cu_out, id_matrix_cu_out, dim_world, dim_filter, number_of_creatures);

    //2: update world value with new value from convolution
    cudaMemcpy( world_cu, world_cu_out, dim_world*dim_world*sizeof(float) ,cudaMemcpyDeviceToDevice );
    cudaMemcpy( id_matrix_cu, id_matrix_cu_out, dim_world*dim_world*sizeof(int) ,cudaMemcpyDeviceToDevice );

    //2: return world result for printing
    cudaMemcpy( world, world_cu, dim_world*dim_world*sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy( id_matrix, id_matrix_cu, dim_world*dim_world*sizeof(int), cudaMemcpyDeviceToHost );

    //2: printing
    printing_world("world after convolution",world,id_matrix,dim_world);

    //end: free memory (fine processo)

    free(world);
    free(id_matrix);
    free(filter);

    cudaFree(world_cu);
    cudaFree(id_matrix_cu);
    cudaFree(filter_cu);
    cudaFree(world_cu_out);
    cudaFree(id_matrix_cu_out);

    float x = -1;
    bool end = !(x+1);
    float ris = 100 * end;

    printf("end: %d, ris: %f \n",end,ris);

}