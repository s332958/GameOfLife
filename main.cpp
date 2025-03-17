#include "libs/loader.h"
#include "libs/kernel.cuh"
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <format>

#define MAX_SAVED_WORLDS 1
//cudaMallocAsync e cudaFreeAsync disponibili solo su GPU con Compute Capability >= 7.0
const int WIDTH = 1024;
const int HEIGHT = 1024;
GLFWwindow* window;
GLuint textureID;


float colori[21][3]; // 21 perché includiamo l'indice 0


void mappaColori(float* mondo, int* id_matrix, float* mondo_rgb, int width, int height) {
    // Inizializza i colori casuali per i valori da 1 a 20
    srand(static_cast<unsigned int>(time(0))); // Inizializza il seed per rand()

    // Mappa i valori di id_matrix ai colori
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            int id = id_matrix[index];

            if (id == -1) {
                // Bianco
                mondo_rgb[index * 3 + 0] = 255.0f; // R
                mondo_rgb[index * 3 + 1] = 0.0f; // G
                mondo_rgb[index * 3 + 2] = 0.0f; // B
            } else if (id == 0) {
                // Nero
                mondo_rgb[index * 3 + 0] = 255.0f*mondo[index]/65025.0f;; // R
                mondo_rgb[index * 3 + 1] = 255.0f*mondo[index]/65025.0f;; // G
                mondo_rgb[index * 3 + 2] = 255.0f*mondo[index]/65025.0f;; // B
            } else if (id >= 1 && id <= 20) {
                // Colori casuali predefiniti
                mondo_rgb[index * 3 + 0] = colori[id][0]*mondo[index]/65025.0f; // R
                mondo_rgb[index * 3 + 1] = colori[id][1]*mondo[index]/65025.0f; // G
                mondo_rgb[index * 3 + 2] = colori[id][2]*mondo[index]/65025.0f; // B
            } else {
                // Valore non valido (ad esempio, nero)
                mondo_rgb[index * 3 + 0] = 0.0f; // R
                mondo_rgb[index * 3 + 1] = 0.0f; // G
                mondo_rgb[index * 3 + 2] = 0.0f; // B
            }
        }
    }
}

void controllo_errore_cuda(const std::string& descrizione, cudaError_t errore){
    if(errore==cudaError::cudaSuccess) return;
    printf("%s: %s\n",descrizione.c_str(),cudaGetErrorString(errore));
}

void simulazione(std::string& world_name, std::vector<std::string>& filters_name, std::vector<std::string>& creature_names, 
                 int id_simulation, std::vector<Posizione> posizioni, int number_of_creatures, int numbers_of_convolution, 
                 cudaStream_t stream, std::ofstream& file_mondo, std::ofstream& file_id_matrix,
                 cudaDeviceProp const &device_properties, bool render){

    cudaError_t err = cudaSuccess;
    const int mem_max = device_properties.totalGlobalMem;
    const int compute_capability = device_properties.major;

    // Dimensioni e matrici per creature, filtro, mondo e id_matrix
    int *dim_creature, dim_mondo, dim_filtro, *id_matrix;
    float **filtri, **creature, *mondo, *mondo_rgb;
    unsigned char *mondo_save, *id_matrix_save;


    creature = (float**) malloc(number_of_creatures*sizeof(float*));
    filtri = (float**) malloc(number_of_creatures*sizeof(float*));
    dim_creature = (int*) malloc(number_of_creatures*sizeof(int));

    
    std::string nome_mondo = world_name;
    readWorld(nome_mondo.c_str(), &dim_mondo, &mondo, &id_matrix, &mondo_rgb);
    
    mondo_save = (unsigned char*) malloc(dim_mondo*dim_mondo*MAX_SAVED_WORLDS*sizeof(unsigned char));
    id_matrix_save = (unsigned char*) malloc(dim_mondo*dim_mondo*MAX_SAVED_WORLDS*sizeof(unsigned char));


    for(int i = 0; i < number_of_creatures; i++){
        std::string nome_creatura = creature_names[i];
        std::string nome_filtro = filters_name[i];
        readMatrix(nome_filtro.c_str(), &dim_filtro, &filtri[i]);
        readMatrix(nome_creatura.c_str(), &dim_creature[i], &creature[i]);
    }

    int numero_creature = 0;
    float *filtro_cu, *creature_cu;
    float *mondo_cu, *mondo_creature;
    float *creature_energy_cu;
    int *id_matrix_cu;
    unsigned char *mondo_cu_save, *id_matrix_cu_save;

    if(compute_capability>=7) controllo_errore_cuda("allocazione mondo", cudaMallocAsync((void**)&mondo_cu, dim_mondo*dim_mondo*sizeof(float),stream));
    else controllo_errore_cuda("allocazione mondo", cudaMalloc((void**)&mondo_cu, dim_mondo*dim_mondo*sizeof(float)));
    if(compute_capability>=7) controllo_errore_cuda("allocazione id_matrix", cudaMallocAsync((void**)&id_matrix_cu, dim_mondo*dim_mondo*sizeof(int),stream));
    else controllo_errore_cuda("allocazione id_matrix", cudaMalloc((void**)&id_matrix_cu, dim_mondo*dim_mondo*sizeof(int)));

    if(compute_capability>=7) controllo_errore_cuda("allocazione mondo", cudaMallocAsync((void**)&mondo_cu_save, dim_mondo*dim_mondo*sizeof(unsigned char)*MAX_SAVED_WORLDS,stream));
    else controllo_errore_cuda("allocazione mondo", cudaMalloc((void**)&mondo_cu_save, dim_mondo*dim_mondo*sizeof(unsigned char)*MAX_SAVED_WORLDS));
    if(compute_capability>=7) controllo_errore_cuda("allocazione id_matrix", cudaMallocAsync((void**)&id_matrix_cu_save, dim_mondo*dim_mondo*sizeof(unsigned char)*MAX_SAVED_WORLDS,stream));
    else controllo_errore_cuda("allocazione id_matrix", cudaMalloc((void**)&id_matrix_cu_save, dim_mondo*dim_mondo*sizeof(unsigned char)*MAX_SAVED_WORLDS));

    controllo_errore_cuda("passaggio mondo su GPU", cudaMemcpyAsync(mondo_cu, mondo, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyHostToDevice, stream));
    controllo_errore_cuda("passaggio id_matrix su GPU", cudaMemcpyAsync(id_matrix_cu, id_matrix, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyHostToDevice, stream));

    for(int i = 0; i < number_of_creatures; i++){
        controllo_errore_cuda("allocazione creatura", cudaMalloc((void**)&creature_cu, dim_creature[i]*dim_creature[i]*sizeof(float)));
        controllo_errore_cuda("passaggio creatura su GPU", cudaMemcpyAsync(creature_cu, creature[i], dim_creature[i]*dim_creature[i]*sizeof(float), cudaMemcpyHostToDevice, stream));
        wrap_add_creature_to_world(creature_cu, mondo_cu, id_matrix_cu, dim_creature[i], dim_mondo, posizioni[i].getX(), posizioni[i].getY(), numero_creature+1, &numero_creature, stream);
        controllo_errore_cuda("liberazione memoria creatura appena allocata in GPU", cudaFree(creature_cu));

    }
    controllo_errore_cuda("Sincronizzazione Stream dopo aggiunta creature",cudaStreamSynchronize(stream));

    //Possibile problema non so se alloca bene i filtri
    controllo_errore_cuda("allocazione filtro", cudaMalloc((void**)&filtro_cu, dim_filtro*dim_filtro*sizeof(float)*number_of_creatures));
    for(int i=0; i<number_of_creatures; i++){
        controllo_errore_cuda("passaggio filtro i su GPU", cudaMemcpyAsync(filtro_cu+(dim_filtro*dim_filtro*i), filtri[i], dim_filtro*dim_filtro*sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    controllo_errore_cuda("Sincronizzazione Stream dopo salvataggio filtri su GPU",cudaStreamSynchronize(stream));
    
    /*
    float *mondo_out_cu;
    int *id_matrix_out_cu;
    
    controllo_errore_cuda("allocazione memoria mondo_out su GPU", cudaMallocAsync((void**)&mondo_out_cu, dim_mondo*dim_mondo*sizeof(float),stream));
    controllo_errore_cuda("allocazione memoria matrice_index_out su GPU", cudaMallocAsync((void**)&id_matrix_out_cu, dim_mondo*dim_mondo*sizeof(int),stream));
    */

    if(compute_capability>=7) controllo_errore_cuda("allocazione mondo creature", cudaMallocAsync((void**)&mondo_creature, number_of_creatures*dim_mondo*dim_mondo*sizeof(float),stream));
    else controllo_errore_cuda("allocazione mondo creature", cudaMalloc((void**)&mondo_creature, number_of_creatures*dim_mondo*dim_mondo*sizeof(float)));

    wrap_add_base_food(mondo_cu, id_matrix_cu, 200, dim_mondo);
    controllo_errore_cuda("Sincronizzazione Stream dopo wrap_add_base_food",cudaStreamSynchronize(stream));

    for( int j= 0; j < numbers_of_convolution; j++){
       double gpu_time_used;
       clock_t start, end;
       for(int i = 0; i < MAX_SAVED_WORLDS; i++){
            start = clock();

            if(render){
                if (glfwWindowShouldClose(window)) {
                    std::cout << "Finestra chiusa. Terminazione del programma." << std::endl;
                    break; // Esce dal ciclo
                }
            }

            wrap_convolution(mondo_creature,mondo_cu, id_matrix_cu, filtro_cu, mondo_cu_save, id_matrix_cu_save, dim_mondo, dim_filtro, numero_creature, i, stream);

            //controllo_errore_cuda("Sincronizzazione Stream dopo convoluzioni",cudaStreamSynchronize(stream));
            controllo_errore_cuda("passaggio mondo su CPU", cudaMemcpyAsync(mondo, mondo_cu, dim_mondo*dim_mondo*sizeof(float), cudaMemcpyDeviceToHost, stream));
            controllo_errore_cuda("passaggio id_matrix su CPU", cudaMemcpyAsync(id_matrix, id_matrix_cu, dim_mondo*dim_mondo*sizeof(int), cudaMemcpyDeviceToHost, stream));

            mappaColori(mondo, id_matrix, mondo_rgb, dim_mondo, dim_mondo);

            if(render){
                // Carica i dati nella texture OpenGL
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, dim_mondo, dim_mondo, 0, GL_RGB, GL_FLOAT, mondo_rgb);

                // Pulizia del buffer di colore
                glClear(GL_COLOR_BUFFER_BIT);

                // Renderizzazione della texture su un quad
                glBegin(GL_QUADS);
                glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
                glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
                glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
                glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
                glEnd();

                // Swap dei buffer
                glfwSwapBuffers(window);

                // Gestione degli eventi
                glfwPollEvents();
            }


            end = clock();
            gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
            printf("%d -- Tempo di esecuzione: %.5f secondi\n",i, gpu_time_used);
        }               
        start = clock();
    
        //controllo_errore_cuda("passaggio mondo_save su CPU", cudaMemcpyAsync(mondo_save, mondo_cu_save, dim_mondo*dim_mondo*MAX_SAVED_WORLDS*sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
        //controllo_errore_cuda("passaggio id_matrix_save su CPU", cudaMemcpyAsync(id_matrix_save, id_matrix_cu_save, dim_mondo*dim_mondo*MAX_SAVED_WORLDS*sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
        
        //save_matrices_to_file(file_mondo, mondo_save, MAX_SAVED_WORLDS, dim_mondo, true);
        //save_matrices_to_file(file_id_matrix, id_matrix_save, MAX_SAVED_WORLDS, dim_mondo, true);
        end = clock();
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("%d -- Tempo di save: %.5f secondi\n",j, gpu_time_used);
    }

    if(render){
        glDeleteTextures(1, &textureID);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    float *creature_value_tot, *creature_value_tot_cu;
    int *creature_occupation, *creature_occupation_cu;
    int n_creature_obstacles = number_of_creatures + 2;

    creature_value_tot = (float*) malloc(sizeof(float)*n_creature_obstacles);
    creature_occupation = (int*) malloc(sizeof(int)*n_creature_obstacles);

    for(int i=0;i<n_creature_obstacles;i++){
        creature_value_tot[i] = 0.0;
        creature_occupation[i] = 0;
        //printf("%d: val: %f occ: %d \n",i,creature_value_tot[i],creature_occupation[i]);
    }

    controllo_errore_cuda("Allocazione memoria creature occupation GPU", cudaMalloc((void**) &creature_occupation_cu,sizeof(int)*n_creature_obstacles));
    controllo_errore_cuda("Allocazione memoria creature tot value GPU", cudaMalloc((void**) &creature_value_tot_cu,sizeof(float)*n_creature_obstacles));

    controllo_errore_cuda("Spostamento creature value tot da CPU a GPU", cudaMemcpyAsync(creature_value_tot_cu,creature_value_tot,sizeof(float)*n_creature_obstacles,cudaMemcpyHostToDevice,stream));
    controllo_errore_cuda("Spostamento creature occupation da CPU a GPU", cudaMemcpyAsync(creature_occupation_cu,creature_occupation,sizeof(int)*n_creature_obstacles,cudaMemcpyHostToDevice,stream));

    wrap_creature_evaluation(mondo_cu,id_matrix_cu,creature_occupation_cu,creature_value_tot_cu,dim_mondo,n_creature_obstacles,stream);
    
    controllo_errore_cuda("Spostamento creature value tot da GPU a CPU", cudaMemcpyAsync(creature_value_tot,creature_value_tot_cu,sizeof(float)*n_creature_obstacles,cudaMemcpyDeviceToHost,stream));
    controllo_errore_cuda("Spostamento creature occupation da GPU a CPU", cudaMemcpyAsync(creature_occupation,creature_occupation_cu,sizeof(int)*n_creature_obstacles,cudaMemcpyDeviceToHost,stream));
    
    controllo_errore_cuda("Sincronizzazione Stream dopo valutazione",cudaStreamSynchronize(stream));
    for(int i=0; i<numero_creature+1;i++){
        if(i==0) std::cout << "Celle vuote " << creature_occupation[i] << " su " << dim_mondo*dim_mondo <<"\n";
        else std::cout << "Creature " << i << ": " << "cell ocupation: " << creature_occupation[i] << " total value: " << creature_value_tot[i] << "\n";
    }

    controllo_errore_cuda("liberazione memoria mondo_out GPU", cudaFree(mondo_cu_save));
    controllo_errore_cuda("liberazione memoria id_matrix_out GPU", cudaFree(id_matrix_cu_save));
    controllo_errore_cuda("liberazione memoria mondo GPU", cudaFree(mondo_cu));
    controllo_errore_cuda("liberazione memoria id_matrix GPU", cudaFree(id_matrix_cu));
    controllo_errore_cuda("liberazione memoria filtro GPU", cudaFree(filtro_cu));
    controllo_errore_cuda("liberazione memoria creature occupation GPU", cudaFree(creature_occupation_cu));
    controllo_errore_cuda("liberazione memoria creature value to GPU", cudaFree(creature_value_tot_cu));

    free(mondo);
    free(id_matrix);
    free(filtri);
    free(creature);
    free(creature_value_tot);
    free(creature_occupation);
}

int main(int argc, char* argv[]) {
    clock_t start = clock();  // Start time

    bool render = false;
    int numero_convoluzioni = 1;
    
    // Controlla se c'è almeno un argomento
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-render") {
            render = true;
        }
        if(arg == "-group_convolution"){
            numero_convoluzioni = std::atoi(argv[i+1]);
        }
        if(arg == "-gc"){
            numero_convoluzioni = std::atoi(argv[i+1]);
        }
    }

    const int MAX_CREATURE = 64;
    for (int i = 1; i <= 20; i++) {
        colori[i][0] = static_cast<float>(rand() % 256); // R
        colori[i][1] = static_cast<float>(rand() % 256); // G
        colori[i][2] = static_cast<float>(rand() % 256); // B
    }
    
    if(render){
        if (!glfwInit()) {
            std::cerr << "Errore nell'inizializzazione di GLFW" << std::endl;
            return -1;
        }
        window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL Image Rendering", NULL, NULL);
    
        if (!window) {
            std::cerr << "Errore nella creazione della finestra" << std::endl;
            glfwTerminate();
            return -1;
        }
    
        glfwMakeContextCurrent(window);
        // Inizializzazione di OpenGL
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    // Leggo le statistiche della mia GPU
    cudaDeviceProp properties;
    cudaGetDeviceProperties_v2(&properties,0);

    // Leggi la configurazione
    std::vector<SimulationSetup> simulationSetup = readConfiguration("data/configurations/configuration.txt");
    int numero_stream = simulationSetup.size();  // Numero stream in base al numero di configurazioni

    // Dichiarazione degli stream CUDA
    cudaStream_t vs[10];  // Numero di stream massimo

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
            simulationSetup[i].creatureFilterListName,
            simulationSetup[i].creatureListNames,  
            i,
            simulationSetup[i].creturesPositions,
            simulationSetup[i].numberCreatures,
            numero_convoluzioni,
            vs[i],
            file_mondo,
            file_id_matrix,
            properties, 
            render
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