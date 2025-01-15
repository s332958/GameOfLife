#include "libs/loader.h"
#include "libs/kernel.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void controllo_errore_cuda(char *descrizione, cudaError_t errore){
    printf("%s: %s\n",descrizione,cudaGetErrorString(errore));
}

int main(){

    cudaError_t err = cudaSuccess;

    int dim_creatura, dim_mondo, dim_filtro;
    float *creatura, *filtro, *mondo;
    int *id_matrix;

    char *nome_mondo = "data/world.txt", *nome_creatura="data/creature.txt", *nome_filtro="data/filter.txt";
    readWorld(nome_mondo,&dim_mondo,&mondo,&id_matrix);
    readMatrix(nome_creatura,&dim_creatura,&creatura);
    readMatrix(nome_filtro,&dim_filtro,&filtro);

    printing_world("Stampa del mondo iniziale",mondo,id_matrix,dim_mondo);
    printing_matrix("Stampa della creatura",creatura,dim_creatura);
    printing_matrix("Stampa del filtro",filtro,dim_filtro);

    int numero_creature = 0;

    float *mondo_cu, *filtro_cu, *creature_cu;
    int *id_matrix_cu;

    controllo_errore_cuda("allocazione mondo",cudaMalloc( (void**)&mondo_cu, dim_mondo*dim_mondo*sizeof(float) ));
    controllo_errore_cuda("allocazione id_matrix",cudaMalloc( (void**)&id_matrix_cu, dim_mondo*dim_mondo*sizeof(int) ));
    controllo_errore_cuda("allocazione filtro",cudaMalloc( (void**)&filtro_cu, dim_filtro*dim_filtro*sizeof(float) ));
    controllo_errore_cuda("allocazione creatura",cudaMalloc( (void**)&creature_cu, dim_creatura*dim_creatura*sizeof(float) ));

    controllo_errore_cuda("passaggio mondo su GPU",cudaMemcpy( mondo_cu, mondo, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyHostToDevice ));
    controllo_errore_cuda("passaggio id_matrix su GPU",cudaMemcpy( id_matrix_cu, id_matrix, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyHostToDevice ));

    controllo_errore_cuda("passaggio filtro su GPU",cudaMemcpy( filtro_cu, filtro, dim_filtro*dim_filtro*sizeof(float), cudaMemcpyHostToDevice ));
    controllo_errore_cuda("passaggio creatura su GPU",cudaMemcpy( creature_cu, creatura, dim_creatura*dim_creatura*sizeof(float), cudaMemcpyHostToDevice ));

    wrap_add_creature_to_world(creature_cu,mondo_cu,id_matrix_cu,dim_creatura,dim_mondo,0,0,numero_creature+1,&numero_creature);

    controllo_errore_cuda("passaggio mondo su CPU",cudaMemcpy( mondo, mondo_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost ));
    controllo_errore_cuda("passaggio id_matrix su CPU",cudaMemcpy( id_matrix, id_matrix_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost ));

    controllo_errore_cuda("passaggio filtro su CPU",cudaMemcpy( filtro, filtro_cu, dim_filtro*dim_filtro*sizeof(float), cudaMemcpyDeviceToHost ));
    controllo_errore_cuda("passaggio creatura su CPU",cudaMemcpy( creatura, creature_cu, dim_creatura*dim_creatura*sizeof(float), cudaMemcpyDeviceToHost ));

    wrap_add_creature_to_world(creature_cu,mondo_cu,id_matrix_cu,dim_creatura,dim_mondo,4,3,numero_creature+1,&numero_creature);
    
    controllo_errore_cuda("passaggio mondo su CPU",cudaMemcpy( mondo, mondo_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost ));
    controllo_errore_cuda("passaggio id_matrix su CPU",cudaMemcpy( id_matrix, id_matrix_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost ));

    controllo_errore_cuda("passaggio filtro su CPU",cudaMemcpy( filtro, filtro_cu, dim_filtro*dim_filtro*sizeof(float), cudaMemcpyDeviceToHost ));
    controllo_errore_cuda("passaggio creatura su CPU",cudaMemcpy( creatura, creature_cu, dim_creatura*dim_creatura*sizeof(float), cudaMemcpyDeviceToHost ));

    printing_world("Stampa del mondo dopo la aggiunta della creatura",mondo,id_matrix,dim_mondo);

    float *mondo_out_cu;
    int *id_matrix_out_cu;

    controllo_errore_cuda("allocazione memoria mondo_out su GPU",cudaMalloc((void **) &mondo_out_cu,dim_mondo*dim_mondo*sizeof(float)));
    controllo_errore_cuda("allocazione memoria matrice_index_out su GPU",cudaMalloc((void **) &id_matrix_out_cu,dim_mondo*dim_mondo*sizeof(int)));

    wrap_convolution(mondo_cu,id_matrix_cu,filtro_cu,mondo_out_cu,id_matrix_out_cu,dim_mondo,dim_filtro,numero_creature);

    controllo_errore_cuda("passaggio world out alla GPU a world_cu",cudaMemcpy( mondo_cu, mondo_out_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToDevice ));
    controllo_errore_cuda("passaggio id matrix alla GPU a id_matrix_cu",cudaMemcpy( id_matrix_cu, id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToDevice ));

    controllo_errore_cuda("passaggio world out alla CPU",cudaMemcpy( mondo, mondo_out_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost ));
    controllo_errore_cuda("passaggio id matrix alla CPU",cudaMemcpy( id_matrix, id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost ));

    printing_world("mondo dopo la convouzione",mondo,id_matrix,dim_mondo);

    wrap_convolution(mondo_cu,id_matrix_cu,filtro_cu,mondo_out_cu,id_matrix_out_cu,dim_mondo,dim_filtro,numero_creature);

    controllo_errore_cuda("passaggio world out alla GPU a world_cu",cudaMemcpy( mondo_cu, mondo_out_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToDevice ));
    controllo_errore_cuda("passaggio id matrix alla GPU a id_matrix_cu",cudaMemcpy( id_matrix_cu, id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToDevice ));

    controllo_errore_cuda("passaggio world out alla CPU",cudaMemcpy( mondo, mondo_out_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost ));
    controllo_errore_cuda("passaggio id matrix alla CPU",cudaMemcpy( id_matrix, id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost ));

    printing_world("mondo dopo la convouzione",mondo,id_matrix,dim_mondo);

}