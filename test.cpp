#include "libs/loader.h"
#include "libs/kernel.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void controllo_errore_cuda(char *descrizione, cudaError_t errore){
    printf("%s: %s\n",descrizione,cudaGetErrorString(errore));
}

void simulazione(char *world_name, char *filter_name, char **creature_names){
    cudaError_t err = cudaSuccess;
    cudaStream_t stream;

    cudaStreamCreate(&stream);
    controllo_errore_cuda("creazione stream simulazione",cudaStreamCreate(&stream));

    int dim_creatura, dim_mondo, dim_filtro;
    float *creatura, *filtro, *mondo;
    int *id_matrix;

    char *nome_mondo = world_name, *nome_creatura=creature_names[0], *nome_filtro=filter_name;
    readWorld(nome_mondo,&dim_mondo,&mondo,&id_matrix);
    readMatrix(nome_creatura,&dim_creatura,&creatura);
    readMatrix(nome_filtro,&dim_filtro,&filtro);

    printing_world("Stampa del mondo iniziale",mondo,id_matrix,dim_mondo);
    printing_matrix("Stampa della creatura",creatura,dim_creatura);
    printing_matrix("Stampa del filtro",filtro,dim_filtro);

    int numero_creature = 0;

    float *mondo_cu, *filtro_cu, *creature_cu;
    int *id_matrix_cu;

    controllo_errore_cuda("allocazione mondo",cudaMallocAsync( (void**)&mondo_cu, dim_mondo*dim_mondo*sizeof(float), stream ));
    controllo_errore_cuda("allocazione id_matrix",cudaMallocAsync( (void**)&id_matrix_cu, dim_mondo*dim_mondo*sizeof(int), stream ));
    controllo_errore_cuda("allocazione filtro",cudaMallocAsync( (void**)&filtro_cu, dim_filtro*dim_filtro*sizeof(float), stream ));
    controllo_errore_cuda("allocazione creatura",cudaMallocAsync( (void**)&creature_cu, dim_creatura*dim_creatura*sizeof(float), stream ));

    controllo_errore_cuda("passaggio mondo su GPU",cudaMemcpyAsync( mondo_cu, mondo, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyHostToDevice, stream ));
    controllo_errore_cuda("passaggio id_matrix su GPU",cudaMemcpyAsync( id_matrix_cu, id_matrix, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyHostToDevice, stream ));

    controllo_errore_cuda("passaggio filtro su GPU",cudaMemcpyAsync( filtro_cu, filtro, dim_filtro*dim_filtro*sizeof(float), cudaMemcpyHostToDevice, stream ));
    controllo_errore_cuda("passaggio creatura su GPU",cudaMemcpyAsync( creature_cu, creatura, dim_creatura*dim_creatura*sizeof(float), cudaMemcpyHostToDevice, stream ));

    wrap_add_creature_to_world(creature_cu,mondo_cu,id_matrix_cu,dim_creatura,dim_mondo,0,0,numero_creature+1,&numero_creature,stream);

    controllo_errore_cuda("passaggio mondo su CPU",cudaMemcpyAsync( mondo, mondo_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost, stream ));
    controllo_errore_cuda("passaggio id_matrix su CPU",cudaMemcpyAsync( id_matrix, id_matrix_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost, stream ));

    controllo_errore_cuda("passaggio filtro su CPU",cudaMemcpyAsync( filtro, filtro_cu, dim_filtro*dim_filtro*sizeof(float), cudaMemcpyDeviceToHost, stream ));
    controllo_errore_cuda("passaggio creatura su CPU",cudaMemcpyAsync( creatura, creature_cu, dim_creatura*dim_creatura*sizeof(float), cudaMemcpyDeviceToHost, stream ));

    printing_world("Stampa del mondo dopo la aggiunta della creatura",mondo,id_matrix,dim_mondo);

    float *mondo_out_cu;
    int *id_matrix_out_cu;

    controllo_errore_cuda("allocazione memoria mondo_out su GPU",cudaMallocAsync((void **) &mondo_out_cu,dim_mondo*dim_mondo*sizeof(float), stream));
    controllo_errore_cuda("allocazione memoria matrice_index_out su GPU",cudaMallocAsync((void **) &id_matrix_out_cu,dim_mondo*dim_mondo*sizeof(int), stream));

    wrap_convolution(mondo_cu,id_matrix_cu,filtro_cu,mondo_out_cu,id_matrix_out_cu,dim_mondo,dim_filtro,numero_creature,stream);

    controllo_errore_cuda("passaggio world out alla GPU a world_cu",cudaMemcpyAsync( mondo_cu, mondo_out_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToDevice, stream ));
    controllo_errore_cuda("passaggio id matrix alla GPU a id_matrix_cu",cudaMemcpyAsync( id_matrix_cu, id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToDevice,stream ));

    controllo_errore_cuda("passaggio world out alla CPU",cudaMemcpyAsync( mondo, mondo_out_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost, stream ));
    controllo_errore_cuda("passaggio id matrix alla CPU",cudaMemcpyAsync( id_matrix, id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost, stream ));

    printing_world("mondo dopo la convouzione",mondo,id_matrix,dim_mondo);

    cudaFreeAsync(&mondo_out_cu,stream);
    cudaFreeAsync(&id_matrix_out_cu,stream);
    cudaFreeAsync(&mondo_cu,stream);
    cudaFreeAsync(&id_matrix_cu,stream);
    cudaFreeAsync(&filtro_cu,stream);
    cudaFreeAsync(&creature_cu,stream);

    free(mondo);
    free(id_matrix);
    free(filtro);
    free(creatura);

    cudaStreamDestroy(stream);

}

int main(){

    char *nome_mondo = "data/world.txt", *nome_creatura="data/creature.txt", *nome_filtro="data/filter.txt";
    char **creature = (char**) malloc(10 * sizeof(char*));
    creature[0] = nome_creatura;

    simulazione(nome_mondo,nome_filtro,creature);
    simulazione(nome_mondo,nome_filtro,creature);
    simulazione(nome_mondo,nome_filtro,creature);

}