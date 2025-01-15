#include "libs/loader.h"
#include "libs/kernel.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void controllo_errore_cuda(char *descrizione, cudaError_t errore){
    printf("%s: %s\n",descrizione,cudaGetErrorString(errore));
}

void simulazione(char *world_name, char *filter_name, char **creature_names, int* vx, int* vy, int number_of_creatures, cudaStream_t stream){
    cudaError_t err = cudaSuccess;

    //controllo_errore_cuda("creazione stream simulazione",cudaStreamCreate(&stream));

    int *dim_creature, dim_mondo, dim_filtro;
    float **creature, *filtro, *mondo;
    int *id_matrix;

    creature = (float**) malloc(number_of_creatures*sizeof(float*));
    dim_creature = (int*) malloc(number_of_creatures*sizeof(int));

    char *nome_mondo = world_name, *nome_creatura=creature_names[0], *nome_filtro=filter_name;
    readWorld(nome_mondo,&dim_mondo,&mondo,&id_matrix);
    readMatrix(nome_filtro,&dim_filtro,&filtro);

    printing_world("Stampa del mondo iniziale",mondo,id_matrix,dim_mondo);
    printing_matrix("Stampa del filtro",filtro,dim_filtro);

    for(int i=0;i<number_of_creatures;i++){
        char *nome_creatura=creature_names[i];
        readMatrix(nome_creatura,&dim_creature[i],&creature[i]);
        printing_matrix("Stampa della creatura",creature[i],dim_creature[i]);
    }

    int numero_creature = 0;

    float *mondo_cu, *filtro_cu, *creature_cu;
    int *id_matrix_cu;

    controllo_errore_cuda("allocazione mondo",cudaMalloc( (void**)&mondo_cu, dim_mondo*dim_mondo*sizeof(float) ));
    controllo_errore_cuda("allocazione id_matrix",cudaMalloc( (void**)&id_matrix_cu, dim_mondo*dim_mondo*sizeof(int) ));

    controllo_errore_cuda("passaggio mondo su GPU",cudaMemcpyAsync( mondo_cu, mondo, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyHostToDevice, stream ));
    controllo_errore_cuda("passaggio id_matrix su GPU",cudaMemcpyAsync( id_matrix_cu, id_matrix, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyHostToDevice, stream ));

    for(int i=0;i<number_of_creatures;i++){

        controllo_errore_cuda("allocazione creatura",cudaMalloc( (void**)&creature_cu, dim_creature[i]*dim_creature[i]*sizeof(float) ));
        controllo_errore_cuda("passaggio creatura su GPU",cudaMemcpyAsync( creature_cu, creature[i], dim_creature[i]*dim_creature[i]*sizeof(float), cudaMemcpyHostToDevice, stream ));
        wrap_add_creature_to_world(creature_cu,mondo_cu,id_matrix_cu,dim_creature[i],dim_mondo,vx[i],vy[i],numero_creature+1,&numero_creature,stream);
        controllo_errore_cuda("liberazione memoria creatura appena allocata in GPU",cudaFree(creature_cu));

    }

    controllo_errore_cuda("passaggio mondo su CPU",cudaMemcpyAsync( mondo, mondo_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost, stream ));
    controllo_errore_cuda("passaggio id_matrix su CPU",cudaMemcpyAsync( id_matrix, id_matrix_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost, stream ));

    printing_world("Stampa del mondo dopo la aggiunta della creatura",mondo,id_matrix,dim_mondo);

    controllo_errore_cuda("allocazione filtro",cudaMalloc( (void**)&filtro_cu, dim_filtro*dim_filtro*sizeof(float) ));
    controllo_errore_cuda("passaggio filtro su GPU",cudaMemcpyAsync( filtro_cu, filtro, dim_filtro*dim_filtro*sizeof(float), cudaMemcpyHostToDevice, stream ));

    float *mondo_out_cu;
    int *id_matrix_out_cu;

    controllo_errore_cuda("allocazione memoria mondo_out su GPU",cudaMalloc((void **) &mondo_out_cu,dim_mondo*dim_mondo*sizeof(float) ));
    controllo_errore_cuda("allocazione memoria matrice_index_out su GPU",cudaMalloc((void **) &id_matrix_out_cu,dim_mondo*dim_mondo*sizeof(int) ));

    wrap_convolution(mondo_cu,id_matrix_cu,filtro_cu,mondo_out_cu,id_matrix_out_cu,dim_mondo,dim_filtro,numero_creature,stream);

    controllo_errore_cuda("passaggio world out alla GPU a world_cu",cudaMemcpyAsync( mondo_cu, mondo_out_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToDevice, stream ));
    controllo_errore_cuda("passaggio id matrix alla GPU a id_matrix_cu",cudaMemcpyAsync( id_matrix_cu, id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToDevice,stream ));

    controllo_errore_cuda("passaggio world out alla CPU",cudaMemcpyAsync( mondo, mondo_out_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost, stream ));
    controllo_errore_cuda("passaggio id matrix alla CPU",cudaMemcpyAsync( id_matrix, id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost, stream ));

    printing_world("mondo dopo la convouzione",mondo,id_matrix,dim_mondo);

    controllo_errore_cuda("liberazione memoria mondo_out GPU",cudaFree(mondo_out_cu));
    controllo_errore_cuda("liberazione memoria id_matrix_out GPU",cudaFree(id_matrix_out_cu));
    controllo_errore_cuda("liberazione memoria mondo GPU",cudaFree(mondo_cu));
    controllo_errore_cuda("liberazione memoria id_matrix GPU",cudaFree(id_matrix_cu));
    controllo_errore_cuda("liberazione memoria filtro GPU",cudaFree(filtro_cu));

    free(mondo);
    free(id_matrix);
    free(filtro);
    free(creature);

    //cudaStreamDestroy(stream);

}

int main(){

    int const MAX_CREATURE = 10;

    char *nome_mondo = "data/world.txt", *nome_creatura="data/creature.txt", *nome_filtro="data/filter.txt";
    char **creature = (char**) malloc(MAX_CREATURE * sizeof(char*));
    int *vx = (int*) malloc(MAX_CREATURE*sizeof(int));
    int *vy = (int*) malloc(MAX_CREATURE*sizeof(int));
    creature[0] = nome_creatura;
    vx[0] = 1;
    vy[0] = 1;
    creature[1] = nome_creatura;
    vx[1] = 4;
    vy[1] = 4;

    cudaStream_t vs[3];

    for(int i=0;i<3;i++){
        controllo_errore_cuda("creazione stream simulazione",cudaStreamCreate(&vs[i]));
        simulazione(nome_mondo,nome_filtro,creature,vx,vy,2,vs[i]);
    }

    for(int i=0;i<3;i++){
        cudaStreamSynchronize(vs[i]);
        cudaStreamDestroy(vs[i]);
    }

}