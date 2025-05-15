#include "kernel_mondo.cu"
#include "kernel_neuralNet.cu"

#include <cuda_runtime.h>
#include <stdio.h>


//================================================================================================


void check_error(char *descrizione){
    cudaError_t errore = cudaGetLastError();
    if(errore!=cudaSuccess){
        printf("Errore cuda in '%s': %s\n",descrizione,cudaGetErrorString(errore));
    }
}



// ================================================================================================

void load_world_on_GPU(
    float *world_val_h, int *world_id_h,
    float **world_val_d, int **world_id_d,
    int dim, cudaStream_t stream, int cc
                    ) {

    int totdim = dim * dim;

    // Alloco memoria sulla GPU
    if(cc >= 7) cudaMallocAsync((void**)world_val_d, totdim * sizeof(float), stream);
    else cudaMalloc((void**)world_val_d, totdim * sizeof(float));
    check_error("Allocazione mondo valori");

    if(cc >= 7) cudaMallocAsync((void**)world_id_d, totdim * sizeof(int), stream);
    else cudaMalloc((void**)world_id_d, totdim * sizeof(int));
    check_error("Allocazione mondo id");

    // Copio dati da host a device
    cudaMemcpyAsync(*world_val_d, world_val_h, totdim * sizeof(float), cudaMemcpyHostToDevice, stream);
    check_error("Caricamento mondo valori");
    
    cudaMemcpyAsync(*world_id_d, world_id_h, totdim * sizeof(int), cudaMemcpyHostToDevice, stream);
    check_error("Caricamento mondo valori");
}




void setup_world(float *mondo_val_d, int *mondo_id_d, int dim_mondo){
    
    // funzione generazione ostacoli
    // funzione generazione del cibo

}

// ===========================================================================================

void add_creatures(
    float *world_value_d, int world_id_d, int dim_world, 
    int n_creature, 
    int cc, cudaStream_t stream
                ){

    // kernel che in base al numero di creature va ad instanziare 

}

// ==================================================================================================

void load_behaviour_on_GPU(
    Creature *creatures, int n_creature, float **weight_d, float **bias_d,
    int cc, cudaStream_t stream
){

    int totdimw = creatures[0].dim_weight * n_creature;
    int totdimb = creatures[0].dim_bias * n_creature;

    if(cc >= 7) cudaMallocAsync((void**)&weight_d, totdimw * sizeof(float), stream);
    else cudaMalloc((void**)&weight_d, totdimw * sizeof(float));
    check_error("Allocazione pesi di tutte le creature creature");

    if(cc >= 7) cudaMallocAsync((void**)&bias_d, totdimb * sizeof(float), stream);
    else cudaMalloc((void**)&bias_d, totdimb * sizeof(float));
    check_error("Allocazione bias di tutte le creature creature");

    for(int i=0; i<n_creature; i++){
        Creature creature = creatures[i];

        cudaMemcpyAsync(*weight_d, creature.weight_model, creature.dim_weight * sizeof(float), cudaMemcpyHostToDevice, stream);
        check_error("Caricamento mondo valori");
        
        cudaMemcpyAsync(*bias_d, creature.bias_model, creature.dim_bias * sizeof(int), cudaMemcpyHostToDevice, stream);
        check_error("Caricamento mondo valori");
    }

}