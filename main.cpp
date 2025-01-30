#include "libs/loader.h"
#include "libs/kernel.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

//cudaMallocAsync e cudaFreeAsync disponibili solo su GPU con Compute Capability >= 7.0

void controllo_errore_cuda(const std::string& descrizione, cudaError_t errore){
    if(errore==cudaError::cudaSuccess) return;
    printf("%s: %s\n",descrizione.c_str(),cudaGetErrorString(errore));
}

void simulazione(std::string& world_name, std::string& filter_name, std::vector<std::string>& creature_names, 
                 int id_simulation, std::vector<Posizione> posizioni, int number_of_creatures, int numbers_of_convolution, 
                 cudaStream_t stream, std::ofstream& file_mondo, std::ofstream& file_id_matrix){

    cudaError_t err = cudaSuccess;

    // Dimensioni e matrici per creature, filtro, mondo e id_matrix
    int *dim_creature, dim_mondo, dim_filtro;
    float **creature, *filtro, *mondo;
    int *id_matrix;

    creature = (float**) malloc(number_of_creatures*sizeof(float*));
    dim_creature = (int*) malloc(number_of_creatures*sizeof(int));

    std::string nome_mondo = world_name;
    std::string nome_filtro = filter_name;
    readWorld(nome_mondo.c_str(), &dim_mondo, &mondo, &id_matrix);
    readMatrix(nome_filtro.c_str(), &dim_filtro, &filtro);

    for(int i = 0; i < number_of_creatures; i++){
        std::string nome_creatura = creature_names[i];
        readMatrix(nome_creatura.c_str(), &dim_creature[i], &creature[i]);
    }

    int numero_creature = 0;

    float *mondo_cu, *filtro_cu, *creature_cu;
    int *id_matrix_cu;

    controllo_errore_cuda("allocazione mondo", cudaMalloc((void**)&mondo_cu, dim_mondo*dim_mondo*sizeof(float)));
    controllo_errore_cuda("allocazione id_matrix", cudaMalloc((void**)&id_matrix_cu, dim_mondo*dim_mondo*sizeof(int)));

    controllo_errore_cuda("passaggio mondo su GPU", cudaMemcpyAsync(mondo_cu, mondo, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyHostToDevice, stream));
    controllo_errore_cuda("passaggio id_matrix su GPU", cudaMemcpyAsync(id_matrix_cu, id_matrix, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyHostToDevice, stream));

    for(int i = 0; i < number_of_creatures; i++){
        controllo_errore_cuda("allocazione creatura", cudaMalloc((void**)&creature_cu, dim_creature[i]*dim_creature[i]*sizeof(float)));
        controllo_errore_cuda("passaggio creatura su GPU", cudaMemcpyAsync(creature_cu, creature[i], dim_creature[i]*dim_creature[i]*sizeof(float), cudaMemcpyHostToDevice, stream));
        wrap_add_creature_to_world(creature_cu, mondo_cu, id_matrix_cu, dim_creature[i], dim_mondo, posizioni[i].getX(), posizioni[i].getY(), numero_creature+1, &numero_creature, stream);
        controllo_errore_cuda("liberazione memoria creatura appena allocata in GPU", cudaFree(creature_cu));

        controllo_errore_cuda("passaggio mondo su CPU", cudaMemcpyAsync(mondo, mondo_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost, stream));
        controllo_errore_cuda("passaggio id_matrix su CPU", cudaMemcpyAsync(id_matrix, id_matrix_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost, stream));

        save_matrix_to_file(file_mondo, mondo, dim_mondo);
        save_matrix_to_file(file_id_matrix, id_matrix, dim_mondo);
    }

    controllo_errore_cuda("allocazione filtro", cudaMalloc((void**)&filtro_cu, dim_filtro*dim_filtro*sizeof(float)));
    controllo_errore_cuda("passaggio filtro su GPU", cudaMemcpyAsync(filtro_cu, filtro, dim_filtro*dim_filtro*sizeof(float), cudaMemcpyHostToDevice, stream));

    float *mondo_out_cu;
    int *id_matrix_out_cu;

    controllo_errore_cuda("allocazione memoria mondo_out su GPU", cudaMalloc((void**)&mondo_out_cu, dim_mondo*dim_mondo*sizeof(float)));
    controllo_errore_cuda("allocazione memoria matrice_index_out su GPU", cudaMalloc((void**)&id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int)));

    for(int i = 0; i < numbers_of_convolution; i++){
        clock_t start, end;
        double gpu_time_used;
        start = clock();

        wrap_convolution(mondo_cu, id_matrix_cu, filtro_cu, mondo_out_cu, id_matrix_out_cu, dim_mondo, dim_filtro, numero_creature, stream);

        controllo_errore_cuda("passaggio world out alla GPU a world_cu", cudaMemcpyAsync(mondo_cu, mondo_out_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        controllo_errore_cuda("passaggio id matrix alla GPU a id_matrix_cu", cudaMemcpyAsync(id_matrix_cu, id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToDevice, stream));

        controllo_errore_cuda("passaggio world out alla CPU", cudaMemcpyAsync(mondo, mondo_out_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost, stream));
        controllo_errore_cuda("passaggio id matrix alla CPU", cudaMemcpyAsync(id_matrix, id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost, stream));

        save_matrix_to_file(file_mondo, mondo, dim_mondo);
        save_matrix_to_file(file_id_matrix, id_matrix, dim_mondo);

        end = clock();
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Tempo di esecuzione: %.5f secondi\n", gpu_time_used);
    }

    controllo_errore_cuda("liberazione memoria mondo_out GPU", cudaFree(mondo_out_cu));
    controllo_errore_cuda("liberazione memoria id_matrix_out GPU", cudaFree(id_matrix_out_cu));
    controllo_errore_cuda("liberazione memoria mondo GPU", cudaFree(mondo_cu));
    controllo_errore_cuda("liberazione memoria id_matrix GPU", cudaFree(id_matrix_cu));
    controllo_errore_cuda("liberazione memoria filtro GPU", cudaFree(filtro_cu));

    free(mondo);
    free(id_matrix);
    free(filtro);
    free(creature);
}

int main() {
    clock_t start = clock();  // Start time

    const int MAX_CREATURE = 64;

    // Leggi la configurazione
    std::vector<SimulationSetup> simulationSetup = readConfiguration("data/configurations/configuration.txt");
    int numero_stream = simulationSetup.size();  // Numero stream in base al numero di configurazioni

    // Dichiarazione degli stream CUDA
    cudaStream_t vs[10];  // Numero di stream massimo
    int numero_convoluzioni = 200;

    std::cout << "Numero di simulazioni: " << numero_stream << std::endl;

    // Creazione degli stream
    for (int i = 0; i < numero_stream; i++) {
        controllo_errore_cuda("creazione stream simulazione", cudaStreamCreate(&vs[i]));

        // Creazione dei file per output
        std::string nome_file_output_mondo = "data/output/mondo" + std::to_string(i) + ".txt";
        std::string nome_file_output_id_matrix = "data/output/id_matrix" + std::to_string(i) + ".txt";
        
        clear_file(nome_file_output_mondo);  // Assicurati che questa funzione sia definita
        clear_file(nome_file_output_id_matrix);

        std::ofstream file_mondo(nome_file_output_mondo, std::ios::app);
        std::ofstream file_id_matrix(nome_file_output_id_matrix, std::ios::app);

        // Chiamata alla simulazione con i parametri corretti
        simulazione(
            simulationSetup[i].worldName,
            simulationSetup[i].filterName,
            simulationSetup[i].creatureListNames,  
            i,
            simulationSetup[i].creturesPositions,
            simulationSetup[i].numberCreatures,
            numero_convoluzioni,
            vs[i],
            file_mondo,
            file_id_matrix
        );

        // Chiusura dei file
        file_mondo.close();
        file_id_matrix.close();
    }

    // Sincronizzazione e distruzione degli stream
    for (int i = 0; i < numero_stream; i++) {
        cudaStreamSynchronize(vs[i]);
        cudaStreamDestroy(vs[i]);
    }

    clock_t end = clock();  // End time
    std::cout << "Tempo esecuzione programma: " << (end - start) / CLOCKS_PER_SEC << " secondi" << std::endl;

    return 0;
}