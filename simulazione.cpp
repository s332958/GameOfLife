
#include "libs/mappa_colori.cuh"
#include "libs/mondo_kernel.cuh"
#include "libs/NN_kernel.cuh"
#include "libs/utils_kernel.cuh"
#include "libs/utils_cpu.h"
#include "libs/perlin_noise.cuh"
#include "simulazione.h"

#include <GLFW/glfw3.h>

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <string>
#define FRAME_TIME_US 16666 

void simulazione(
    Simulation_setup simulation_setup,
    GLFWwindow* window, GLuint textureID
) {


    //  -----------------------------------------
    //  Define Simulation Parameters
    //  -----------------------------------------

    int world_dim = simulation_setup.world_dim;
    int n_creature = simulation_setup.n_creature;
    int const n_layer = simulation_setup.n_layer;
    int model_structure[n_layer] = {0}; 
    std::copy(simulation_setup.model_structure.begin(), simulation_setup.model_structure.end(), model_structure);

    size_t reserve_free_memory = simulation_setup.reserve_free_memory;
    float *weights_models = simulation_setup.weights_models;
    float *biases_models = simulation_setup.biases_models;
    int N_EPOCH = simulation_setup.N_EPOCH;
    int N_STEPS = simulation_setup.N_STEPS;
    int MAX_WORKSPACE = simulation_setup.MAX_WORKSPACE;
    int METHOD_EVAL = simulation_setup.METHOD_EVAL;
    bool render = simulation_setup.render;

    int checkpoint_epoch = simulation_setup.checkpoint_epoch;
    
    float PN_scale_obstacles = simulation_setup.PN_scale_obstacles;
    float PN_threshold_obstacles = simulation_setup.PN_threshold_obstacles;

    float PN_scale_food = simulation_setup.PN_scale_food;
    float PN_threshold_food = simulation_setup.PN_threshold_food;
    float random_threshold_food = simulation_setup.random_threshold_food;
    
    float starting_value = simulation_setup.starting_value;
    float energy_fraction = simulation_setup.energy_fraction;
    float energy_decay = simulation_setup.energy_decay;

    float winners_fraction = simulation_setup.winners_fraction;
    float recombination_newborns_fraction = simulation_setup.recombination_newborns_fraction;
    
    float gen_x_block = simulation_setup.gen_x_block;
    float mutation_probability = simulation_setup.mutation_probability;
    float mutation_range = simulation_setup.mutation_range;

    int clean_window_size = simulation_setup.clean_window_size;

    char path_save_file[300];
    sprintf(path_save_file,"models/%s",simulation_setup.file_model);

    struct timespec start;
    struct timespec end;

    unsigned long seed = (unsigned long)time(NULL);

    // -------------------------------------------
    // PRE-FASE : Info GPU
    // -------------------------------------------

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int cc_major = prop.major;

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "---- Info CUDA ----\n";
    std::cout << "Device ID: " << device << "\n";
    std::cout << "GPU Name: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memory Free: " << (free_mem / (1024 * 1024)) << " MB\n";
    std::cout << "Memory Total: " << (total_mem / (1024 * 1024)) << " MB\n";
    std::cout << "-------------------\n";
    std::cout << "Inizio simulazione...\n";

    // -------------------------------------------
    // PRE-FASE : Inizializzazione generale
    // -------------------------------------------

    // Calcolo dimensione pesi e bias totali e dim kernel visione
    int n_weight = 0;
    int n_bias = 0;
    for(int i = 0; i < n_layer - 1; ++i) {
        n_weight += model_structure[i] * model_structure[i + 1];
        n_bias += model_structure[i + 1];
    }

    size_t tot_world_dim_size_float = sizeof(float) * world_dim * world_dim;
    size_t tot_world_dim_size_int = sizeof(int) * world_dim * world_dim;
    size_t tot_matrix_contribution_size = tot_world_dim_size_float * n_creature;
    size_t tot_energy_vector_size = n_creature * sizeof(float);
    size_t tot_occupation_vector_size = n_creature * sizeof(int);
    size_t tot_models_weight_size = n_creature * n_weight * sizeof(float);
    size_t tot_models_bias_size = n_creature * n_bias * sizeof(float);

    // Allocazioni CPU
    float *world_rgb_h         = (float*) malloc(tot_world_dim_size_float*3);
    float *world_value_h       = (float*) malloc(tot_world_dim_size_float);    
    float *world_signal_h      = (float*) malloc(tot_world_dim_size_float);
    int   *world_id_h          = (int*)   malloc(tot_world_dim_size_int);
    float *energy_vector_h     = (float*) malloc(tot_energy_vector_size);
    float *occupation_vector_h = (float*) malloc(tot_occupation_vector_size);
    int   *creature_ordered_h  = (int*)   malloc(tot_occupation_vector_size);
    int   *alive_cells_h       = (int*)   malloc(tot_world_dim_size_int);
    int   *n_cell_alive_h      ; 
    float *model_weights_h     = nullptr;
    float *model_biases_h      = nullptr;

    // Siccome n_alive_cell è solo un int e viene passato spesso lo alloco nella memoria pinnata della RAM che viene condivsa con la GPU
    cudaHostAlloc((void**)&n_cell_alive_h, sizeof(int), cudaHostAllocMapped);

    if(weights_models==nullptr) model_weights_h = (float*) malloc(tot_models_weight_size);
    else model_weights_h       = weights_models;
    if(biases_models==nullptr)  model_biases_h = (float*) malloc(tot_models_bias_size);
    else model_biases_h        = biases_models;



    // Allocazioni GPU 
    float *world_rgb_d                = (float*) cuda_allocate(tot_world_dim_size_float * 3, cc_major, 0);
    float *world_value_d              = (float*) cuda_allocate(tot_world_dim_size_float, cc_major, 0);    
    float *world_signal_d             = (float*) cuda_allocate(tot_world_dim_size_float, cc_major, 0);
    int   *world_id_d                 = (int*)   cuda_allocate(tot_world_dim_size_int, cc_major, 0);    
    float *world_contributions_d      = (float*) cuda_allocate(tot_matrix_contribution_size, cc_major, 0);
    float *model_weights_d            = (float*) cuda_allocate(tot_models_weight_size, cc_major, 0);
    float *model_biases_d             = (float*) cuda_allocate(tot_models_bias_size, cc_major, 0);
    int   *alive_cells_d              = (int*)   cuda_allocate(tot_world_dim_size_int, cc_major, 0);
    float *energy_vector_d            = (float*) cuda_allocate(tot_energy_vector_size, cc_major, 0);
    float *occupation_vector_d        = (float*) cuda_allocate(tot_occupation_vector_size, cc_major, 0);
    float *new_model_weights_d        = (float*) cuda_allocate(tot_models_weight_size, cc_major, 0);
    float *new_model_biases_d         = (float*) cuda_allocate(tot_models_bias_size, cc_major, 0);    
    int   *support_vector_d           = (int*)   cuda_allocate(tot_world_dim_size_int, cc_major, 0);
    int   *n_cell_alive_d             ; 
    
    cudaHostGetDevicePointer(&n_cell_alive_d, n_cell_alive_h , 0);

    // Find max dim layer
    int i_max = 0;
    for(int i=0; i<n_layer; i++){
        if(model_structure[i]>model_structure[i_max]){
            i_max = i;
        }
    }
    
    // Calcolo spazio massimo disponibile per workspace
    int dim_max_layer = model_structure[i_max];
    int dim_input = model_structure[0];
    int dim_output = model_structure[n_layer - 1];
    size_t workspace_size = dim_max_layer * sizeof(float);

    // compute n_workspace using all free memory avaible
    computeFreeMemory(&free_mem);
    int n_workspace = (free_mem > reserve_free_memory) ? (free_mem - reserve_free_memory) / (workspace_size * 2) : 0;

    // if are not allocate workspace return error
    if (n_workspace == 0) {
        throw std::runtime_error("No memory for workspace... impossible to continue.");
    } else if (n_workspace > world_dim * world_dim) {
        n_workspace = world_dim * world_dim;
    }
    //if(n_workspace>MAX_WORKSPACE) n_workspace=MAX_WORKSPACE;

    std::cout << "Free memory: " << free_mem << " bytes\n";
    std::cout << "Reserved memory: " << reserve_free_memory << " bytes\n";
    std::cout << "Workspace size per slot: " << workspace_size << " bytes\n";
    std::cout << "Allocate " << n_workspace << " workspace slots for the simulation.\n";

    // allocazione workspace
    float *workspace_input_d  = (float*) cuda_allocate(n_workspace * workspace_size, cc_major, 0);
    float *workspace_output_d  = (float*) cuda_allocate(n_workspace * workspace_size, cc_major, 0);
  
    computeFreeMemory(&free_mem);
    std::cout << "Free memory after allocation " << free_mem/1024 << " KB\n";

    printf("END ALLOCATIONS\n");

    // CARIMENTO DATI
    if(biases_models==nullptr && weights_models==nullptr){       
        
        // Generation random models
        launch_fill_random_kernel(model_weights_d,0,n_weight*n_creature,-1.0f,1.0f,seed, 0);
        CUDA_CHECK(cudaGetLastError());
        launch_fill_random_kernel(model_biases_d,0,n_bias*n_creature,-1.0f,1.0f,seed+1, 0);
        CUDA_CHECK(cudaGetLastError());
        
        // Load on CPU vettore pesi tutti i modelli (world_weights_h è vettore host con dati)
        cuda_memcpy(model_weights_h, model_weights_d, tot_models_weight_size, cudaMemcpyDeviceToHost, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        // Load on CPU vettore bias tutti i modelli (world_biases_h è vettore host con dati)
        cuda_memcpy(model_biases_h, model_biases_d, tot_models_bias_size, cudaMemcpyDeviceToHost, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        
        save_model_on_file(path_save_file,model_structure,n_layer,model_weights_h,model_biases_h,n_weight,n_bias,n_creature);

        printf("FINE GENERAZIONE MODELLI \n");
    }
    else if(weights_models!=nullptr && biases_models!=nullptr){
        // Load on GPU vettore pesi tutti i modelli (world_weights_h è vettore host con dati)
        cuda_memcpy(model_weights_d, model_weights_h, tot_models_weight_size, cudaMemcpyHostToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        // Load on GPU vettore bias tutti i modelli (world_biases_h è vettore host con dati)
        cuda_memcpy(model_biases_d, model_biases_h, tot_models_bias_size, cudaMemcpyHostToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
    }

    printf("LOAD MODEL ON GPU\n");
    
        // -------------------------------------------
        // FASE 1 : preparazione epoca 
        // -------------------------------------------
    for (int epoca = 0; epoca < N_EPOCH; epoca++) {
        std::cout << "=======================  Epoca: " << epoca << "  ========================\n";



        
        //std::memset(world_value_h, 0, tot_world_dim_size_float);
        //std::memset(world_id_h, 0, tot_world_dim_size_int);
        std::memset(alive_cells_h, 0, tot_world_dim_size_int);

        cudaMemset(world_signal_d, 0, tot_world_dim_size_float);    
        CUDA_CHECK(cudaGetLastError());    
        cudaMemset(energy_vector_d, 0, tot_energy_vector_size);
        CUDA_CHECK(cudaGetLastError());
        cudaMemset(occupation_vector_d, 0, tot_occupation_vector_size);
        CUDA_CHECK(cudaGetLastError());
        cudaMemset(world_value_d, 0, tot_world_dim_size_float);
        CUDA_CHECK(cudaGetLastError());
        cudaMemset(world_id_d, 0, tot_world_dim_size_int);
        CUDA_CHECK(cudaGetLastError());

        printf("RESET ALL MATRIX \n"); 

                // - Aggiunta ostacoli perlin al mondo
        launch_perlinNoise_obstacles(world_dim, world_id_d, PN_scale_obstacles, PN_threshold_obstacles, 0);
        CUDA_CHECK(cudaGetLastError());
        
        // - Aggiunta cibo perlin al mondo
        launch_perlinNoise_food(world_dim, world_id_d, world_value_d, PN_scale_food, PN_threshold_food, 0);
        CUDA_CHECK(cudaGetLastError());

        cuda_memcpy(world_value_h, world_value_d, tot_world_dim_size_float, cudaMemcpyDeviceToHost, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        cuda_memcpy( world_id_h, world_id_d, tot_world_dim_size_int, cudaMemcpyDeviceToHost, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());

        cudaDeviceSynchronize();

        // ======================================== Aggiunta creature al mondo 

        int random_index = rand() % world_dim*world_dim;
        for (int i = 0; i < n_creature; i++){
            while(world_id_h[random_index] != 0){
                random_index = rand() % (world_dim*world_dim);
            }
            if(world_id_h[random_index] == 0 && world_value_h[random_index] == 0){
                world_value_h[random_index] = starting_value;
                world_id_h[random_index] = i + 1;
                alive_cells_h[i] = random_index;                
                random_index = rand() % (world_dim*world_dim);
            }     
        }

        *n_cell_alive_h = n_creature;

        printf("SETUP CELL ALIVE \n");

        // - Passo il mondo valori, cellule vive ed ID sulla GPU
        cuda_memcpy(world_value_d, world_value_h, tot_world_dim_size_float, cudaMemcpyHostToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        cuda_memcpy(world_id_d, world_id_h, tot_world_dim_size_int, cudaMemcpyHostToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        cuda_memcpy(alive_cells_d, alive_cells_h, tot_world_dim_size_int, cudaMemcpyHostToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());

        
        

        

        // - Aggiunta cibo al mondo
        // - Possibile ottimizzazione togliendo i curandstate
        launch_add_objects_to_world(world_value_d, world_id_d, world_dim, 0, 1.0f, 10.0f, random_threshold_food, 0);
        CUDA_CHECK(cudaGetLastError());

        launch_clean_around_cells(world_value_d, world_id_d, world_dim, alive_cells_d, n_cell_alive_h, clean_window_size, 0);
        CUDA_CHECK(cudaGetLastError());

        cudaDeviceSynchronize();
            // -------------------------------------------
            // FASE 2 : calcolo step 
            // -------------------------------------------
        /*======================================================================================================================================*/
        
        printf(" alive_cell: %d" , *n_cell_alive_h );

        for(int step=0; step<N_STEPS && *n_cell_alive_h > 0; step++){
            
            clock_gettime(CLOCK_MONOTONIC, &start); 
            if(render){
                if (glfwWindowShouldClose(window)) {
                    std::cout << "Finestra chiusa. Terminazione del programma." << std::endl;
                    goto fine;
                }
            }            


            int offset=0;
            int vision = sqrt(dim_input/2);

            cudaMemset(world_contributions_d, 0, tot_matrix_contribution_size);
            CUDA_CHECK(cudaGetLastError());
            cudaMemset(workspace_input_d, 0, n_workspace * workspace_size);
            CUDA_CHECK(cudaGetLastError());
            cudaMemset(workspace_output_d, 0, n_workspace * workspace_size);
            CUDA_CHECK(cudaGetLastError());
            

            int offset_alive_cell = 0;
            int offset_workspace = 0;
            int limit_workspace_cell = n_workspace;
            if(n_workspace > *n_cell_alive_h){
                limit_workspace_cell = *n_cell_alive_h;
            }
            while(offset_workspace<*n_cell_alive_h){               
                
                if(*n_cell_alive_h - offset_workspace < n_workspace){
                    limit_workspace_cell = *n_cell_alive_h - offset_workspace;
                }


                launch_vision(  
                    world_value_d,
                    world_id_d,
                    world_signal_d,
                    world_dim,
                    alive_cells_d+offset_workspace,
                    vision,
                    workspace_input_d,
                    limit_workspace_cell,
                    0
                );
                CUDA_CHECK(cudaGetLastError());




                launch_NN_forward(
                    workspace_input_d,
                    workspace_output_d,
                    workspace_size,
                    model_weights_d,
                    n_weight,
                    model_biases_d,
                    n_bias,
                    model_structure,
                    limit_workspace_cell,
                    alive_cells_d + offset_workspace,
                    world_id_d,
                    n_layer,
                    0
                );
                CUDA_CHECK(cudaGetLastError());  


                launch_output_elaboration(
                    world_value_d,
                    world_signal_d,
                    world_id_d,
                    world_contributions_d,
                    workspace_output_d,
                    alive_cells_d + offset_workspace,
                    world_dim,
                    n_creature,
                    dim_output,
                    limit_workspace_cell,
                    energy_fraction,
                    0
                );
                CUDA_CHECK(cudaGetLastError());

                offset_workspace += limit_workspace_cell;
                //cudaDeviceSynchronize(); //fondamentale


            }
            //cudaDeviceSynchronize();
            launch_world_update(
                world_value_d,
                world_id_d,
                world_signal_d,
                world_contributions_d,
                world_dim,
                n_creature,
                energy_decay,
                0
            ); 
            
            //cudaDeviceSynchronize();        
            
            int new_n_cell = 0;
            compact_with_thrust(world_id_d, alive_cells_d, world_dim, new_n_cell);
            *n_cell_alive_h = new_n_cell;
            
            
            /*
            launch_find_index_cell_alive(
                world_id_d,
                world_dim*world_dim,
                alive_cells_d,
                n_cell_alive_d,
                n_cell_alive_h,
                support_vector_d,
                0
            );
            CUDA_CHECK(cudaGetLastError());
            */


            //VERSIONE CPU
            /*
            launch_cell_alive_check(
                alive_cells_d,
                n_cell_alive_h,
                world_id_d,
                0
            );    
            cuda_memcpy(alive_cells_h, alive_cells_d, tot_world_dim_size_int, cudaMemcpyDeviceToHost, cc_major, 0);
            CUDA_CHECK(cudaGetLastError());  
            int local_cellN = *n_cell_alive_h;
            int counter = 0;
            
            for(int i = 0; i < local_cellN; i++){
                if(alive_cells_h[i] >= 0){
                    alive_cells_h[counter] = alive_cells_h[i];
                    //printf("%d  ",alive_cells_h[counter]);
                    counter += 1;
                }
            }
            *n_cell_alive_h = counter;
            
            cuda_memcpy(alive_cells_d, alive_cells_h, tot_world_dim_size_int, cudaMemcpyHostToDevice, cc_major, 0);
            CUDA_CHECK(cudaGetLastError());
                        // - Ritorno alive_cell_d
            cuda_memcpy(alive_cells_h, alive_cells_d, tot_world_dim_size_int, cudaMemcpyDeviceToHost, cc_major, streams[0]);
            CUDA_CHECK(cudaGetLastError());   
            std::cout << "\n=== ALIVE CELLS ===\n";
            for (int i = 0; i < *n_cell_alive_h; i++) {
                std::cout << "Alive[" << i << "] = " << alive_cells_h[i] << "\n";
                std::cout << " (" << world_id_h[alive_cells_h[i]] << ")\n";

            }
            */

            if(render){

                launch_mappa_colori(world_value_d, world_id_d, world_rgb_d, world_dim, 0);
                CUDA_CHECK(cudaGetLastError());

                //launch_mappa_signal(world_value_d, world_id_d, world_signal_d, world_rgb_d, world_dim, n_creature, 0);
                //CUDA_CHECK(cudaGetLastError());

                cuda_memcpy(world_rgb_h, world_rgb_d, tot_world_dim_size_float * 3, cudaMemcpyDeviceToHost, cc_major, 0);
                CUDA_CHECK(cudaGetLastError());

                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, world_dim, world_dim, GL_RGB, GL_FLOAT, world_rgb_h);
                glClear(GL_COLOR_BUFFER_BIT);

                glBegin(GL_QUADS);
                glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
                glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
                glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
                glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
                glEnd();

                glfwSwapBuffers(window);
                glfwPollEvents();              

            } 

            

            // ======================================== Aggiornamento vettori valutazione occupazione ed energia
            launch_compute_energy_and_occupation(world_value_d,world_id_d,occupation_vector_d,energy_vector_d,world_dim,n_creature, 0);
            CUDA_CHECK(cudaGetLastError());

            clock_gettime(CLOCK_MONOTONIC, &end); 
            char epocstep[64];
            snprintf(epocstep, sizeof(epocstep), "%d.%d", epoca,step);
                        
            long elapsed_us = (end.tv_sec - start.tv_sec) * 1000000 +
                            (end.tv_nsec - start.tv_nsec) / 1000;

            //printf("Step: %10s \t alive_cell: %8d  |  %3.1f it/s \n",epocstep,*n_cell_alive_h,1.0f/((float)(end - start) / CLOCKS_PER_SEC));
            /*
            
            if (elapsed_us < FRAME_TIME_US) {
                usleep(FRAME_TIME_US - elapsed_us);
            }
                */
        }

        // -------------------------------------------
        // FASE 3 : generazione nuove creature 
        // -------------------------------------------

        seed = (unsigned long)time(NULL);

        int limit_winner = n_creature * winners_fraction;
        int limit_recombination = n_creature * recombination_newborns_fraction;

        cuda_memcpy(energy_vector_h,energy_vector_d,tot_energy_vector_size,cudaMemcpyDeviceToHost,cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        cuda_memcpy(occupation_vector_h,occupation_vector_d,tot_occupation_vector_size,cudaMemcpyDeviceToHost,cc_major, 0);
        CUDA_CHECK(cudaGetLastError());

        float tot_score = 0;
        if(METHOD_EVAL==0){
            tot_score = argsort_bubble(energy_vector_h,creature_ordered_h,n_creature);
        } 
        if(METHOD_EVAL==1){
            tot_score = argsort_bubble(occupation_vector_h,creature_ordered_h,n_creature); 
        } 
        
        tot_score = tot_score / n_creature;
        append_score_to_file("log_score.txt", tot_score);

        int first_ID = creature_ordered_h[0];
        if (first_ID < 0 || first_ID >= n_creature) {
            std::cerr << "Errore: first_ID fuori range: " << first_ID << "\n";
        }
        // ======================================== il primo N sempre
        for(int i = 0; i < limit_winner; i++){
            cuda_memcpy(new_model_weights_d, model_weights_d + n_weight * creature_ordered_h[0], n_weight * sizeof(float), cudaMemcpyDeviceToDevice, cc_major, 0);
            CUDA_CHECK(cudaGetLastError());        
            cuda_memcpy(new_model_biases_d, model_biases_d + n_bias * creature_ordered_h[0], n_bias * sizeof(float), cudaMemcpyDeviceToDevice, cc_major, 0);
            CUDA_CHECK(cudaGetLastError());      
        }
        // ======================================== Creazione nuove creature 
        for(int i=1;i<limit_recombination;i++){
            int idx1 = get_random_int(0,limit_winner);
            int idx2 = get_random_int(0,limit_winner);
            int gen1 = creature_ordered_h[idx1];
            int gen2 = creature_ordered_h[idx2];
            
            launch_recombine_models_kernel(
                model_weights_d,
                model_biases_d,
                new_model_weights_d,
                new_model_biases_d,
                n_weight,
                n_bias, 
                gen1, 
                gen2,
                i, 
                gen_x_block, 
                mutation_probability, 
                mutation_range, 
                seed, 
                0
            );
            CUDA_CHECK(cudaGetLastError());         
        }              
        
        // ======================================== Gli ultimi totalmente casuali
        launch_fill_random_kernel(new_model_weights_d,n_weight*limit_recombination,n_weight*n_creature,-1.0f,1.0f,seed + 1, 0);
        CUDA_CHECK(cudaGetLastError());
        launch_fill_random_kernel(new_model_biases_d,n_bias*limit_recombination,n_bias*n_creature,-1.0f,1.0f,seed + 2, 0);
        CUDA_CHECK(cudaGetLastError());              
        
        cuda_memcpy(model_weights_d, new_model_weights_d, tot_models_weight_size, cudaMemcpyDeviceToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        cuda_memcpy(model_biases_d, new_model_biases_d, tot_models_bias_size, cudaMemcpyDeviceToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());

        // ======================================== Salvataggio nuovi modelli
        if((epoca != 0 && epoca%checkpoint_epoch == 0) || epoca == (N_EPOCH - 1)){            
            cuda_memcpy(model_weights_h, new_model_weights_d, tot_models_weight_size, cudaMemcpyDeviceToHost, cc_major, 0);
            CUDA_CHECK(cudaGetLastError());
            cuda_memcpy(model_biases_h, new_model_biases_d, tot_models_bias_size, cudaMemcpyDeviceToHost, cc_major, 0);
            CUDA_CHECK(cudaGetLastError());

            save_model_on_file(path_save_file,model_structure,n_layer,model_weights_h,model_biases_h,n_weight,n_bias,n_creature);
            printf("MODEL GENERATE AND SAVE \n");
        }       

    }


    // -------------------------------------------
    // POST-FASE : 
    // -------------------------------------------
    fine:

    if(render){
        glDeleteTextures(1, &textureID);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    // ======================================== Free zone di memoria GPU
    cuda_Free(world_rgb_d, cc_major, 0);
    cuda_Free(world_value_d, cc_major, 0);
    cuda_Free(world_id_d, cc_major, 0);
    cuda_Free(world_contributions_d, cc_major, 0);
    cuda_Free(world_signal_d, cc_major, 0);
    cuda_Free(alive_cells_d, cc_major, 0);
    cuda_Free(occupation_vector_d, cc_major, 0);
    cuda_Free(energy_vector_d, cc_major, 0);
    cuda_Free(model_biases_d, cc_major, 0);
    cuda_Free(model_weights_d, cc_major, 0);
    cuda_Free(new_model_weights_d, cc_major, 0);
    cuda_Free(new_model_biases_d, cc_major, 0);
    cuda_Free(workspace_input_d, cc_major, 0);

    free(energy_vector_h);
    free(occupation_vector_h);
    free(creature_ordered_h);
    free(model_weights_h);
    free(model_biases_h);

    free(alive_cells_h);
    free(world_id_h);
    free(world_signal_h);
    free(world_rgb_h);
    free(world_value_h);

    cudaFreeHost(n_cell_alive_h);
    
    
    printf("FREE DONE\n");
    

    std::cout << "End Simulation. \n";
}
