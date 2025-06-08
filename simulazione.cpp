
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

    clock_t start;
    clock_t end;

    unsigned long seed = (unsigned long)time(NULL);

    float std = simulation_setup.std;
    float alpha = simulation_setup.alpha;

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
    size_t tot_weights_size = n_weight * sizeof(float);
    size_t tot_biases_size = n_bias * sizeof(float);

    // Allocazioni CPU
    float *world_rgb_h         = (float*) malloc(tot_world_dim_size_float*3);
    float *world_value_h       = (float*) malloc(tot_world_dim_size_float);    
    float *world_signal_h      = (float*) malloc(tot_world_dim_size_float);
    int   *world_id_h          = (int*)   malloc(tot_world_dim_size_int);
    float *energy_vector_h     = (float*) malloc(tot_energy_vector_size);
    float *occupation_vector_h = (float*) malloc(tot_occupation_vector_size);
    int   *alive_cells_h       = (int*)   malloc(tot_world_dim_size_int);
    int   *n_cell_alive_h      ; 
    float *model_weights_h     = nullptr;
    float *model_biases_h      = nullptr;

    // Siccome n_alive_cell è solo un int e viene passato spesso lo alloco nella memoria pinnata della RAM che viene condivsa con la GPU
    cudaHostAlloc((void**)&n_cell_alive_h, sizeof(int), cudaHostAllocMapped);

    // Anche from n model to load in one singel model
    if(weights_models==nullptr) model_weights_h = (float*) malloc(tot_weights_size);
    else model_weights_h       = weights_models;
    if(biases_models==nullptr)  model_biases_h = (float*) malloc(tot_biases_size);
    else model_biases_h        = biases_models;



    // Allocazioni GPU 
    float *world_rgb_d                = (float*) cuda_allocate(tot_world_dim_size_float * 3, cc_major, 0);
    float *world_value_d              = (float*) cuda_allocate(tot_world_dim_size_float, cc_major, 0);    
    float *world_signal_d             = (float*) cuda_allocate(tot_world_dim_size_float, cc_major, 0);
    int   *world_id_d                 = (int*)   cuda_allocate(tot_world_dim_size_int, cc_major, 0);    
    float *world_contributions_d      = (float*) cuda_allocate(tot_matrix_contribution_size, cc_major, 0);
    float *model_weights_d            = (float*) cuda_allocate(tot_weights_size, cc_major, 0);
    float *model_biases_d             = (float*) cuda_allocate(tot_biases_size, cc_major, 0);
    int   *alive_cells_d              = (int*)   cuda_allocate(tot_world_dim_size_int, cc_major, 0);
    float *energy_vector_d            = (float*) cuda_allocate(tot_energy_vector_size, cc_major, 0);
    float *occupation_vector_d        = (float*) cuda_allocate(tot_occupation_vector_size, cc_major, 0);
    float *varation_model_weights_d   = (float*) cuda_allocate(tot_models_weight_size, cc_major, 0);
    float *varation_model_biases_d    = (float*) cuda_allocate(tot_models_bias_size, cc_major, 0);    
    float *new_models_weights_d       = (float*) cuda_allocate(tot_models_weight_size, cc_major, 0);
    float *new_models_biases_d        = (float*) cuda_allocate(tot_models_bias_size, cc_major, 0);
    int   *support_vector_d           = (int*)   cuda_allocate(tot_world_dim_size_int, cc_major, 0);
    curandState *curandStates         = nullptr;
    int   *n_cell_alive_d             ; 
    
    cudaHostGetDevicePointer(&n_cell_alive_d, n_cell_alive_h , 0);
    cudaMalloc((void**) &curandStates, sizeof(curandState)*1024);

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

    // Preparazione curandStates
    launch_init_curandstates(
        curandStates,
        1024,
        0,
        0
    );

    // CARIMENTO DATI
    if(biases_models==nullptr && weights_models==nullptr){       
        
        // Generation random models
        launch_fill_random_kernel(model_weights_d,0,n_weight,-std,std,seed,0);
        CUDA_CHECK(cudaGetLastError());
        launch_fill_random_kernel(model_biases_d,0,n_bias,-std,std,seed+1,0);
        CUDA_CHECK(cudaGetLastError());
        
        // Load on CPU vettore pesi del modello 
        cuda_memcpy(model_weights_h, model_weights_d, tot_weights_size, cudaMemcpyDeviceToHost, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        // Load on CPU vettore bias del modello 
        cuda_memcpy(model_biases_h, model_biases_d, tot_biases_size, cudaMemcpyDeviceToHost, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        
        save_model_on_file(path_save_file,model_weights_h,model_biases_h,n_weight,n_bias);

        printf("FINE GENERAZIONE MODELLI \n");
    }
    else if(weights_models!=nullptr && biases_models!=nullptr){
        // Load on GPU vettore pesi del modello 
        cuda_memcpy(model_weights_d, model_weights_h, tot_weights_size, cudaMemcpyHostToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        // Load on GPU vettore bias del modello 
        cuda_memcpy(model_biases_d, model_biases_h, tot_biases_size, cudaMemcpyHostToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
    }

    printf("LOAD MODEL ON GPU\n");
    
    // -------------------------------------------
    // FASE 1 : preparazione epoca 
    // -------------------------------------------
    for (int epoca = 0; epoca < N_EPOCH; epoca++) {
        std::cout << "=======================  Epoca: " << epoca << "  ========================\n";

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
            while(world_id_h[random_index] != 0 || world_value_h[random_index] != 0){
                random_index = rand() % (world_dim*world_dim);
            }
            world_value_h[random_index] = starting_value;
            world_id_h[random_index] = i + 1;
            alive_cells_h[i] = random_index;                
            random_index = rand() % (world_dim*world_dim); 
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
        


        // - Generazione nuove creature per la simulazione
        launch_generate_clone_creature(
            model_weights_d,
            model_biases_d,
            new_models_weights_d,
            new_models_biases_d,
            varation_model_weights_d,
            varation_model_biases_d,
            n_weight,
            n_bias,
            n_creature,
            std,
            0,
            curandStates
        );

        cudaDeviceSynchronize();
        // -------------------------------------------
        // FASE 2 : calcolo step 
        // -------------------------------------------
        /*======================================================================================================================================*/

        for(int step=0; step<N_STEPS && *n_cell_alive_h > 0; step++){

            // printf(" epoch: %d step: %d alive_cell: %d \n" , epoca, step, *n_cell_alive_h );
            
            start = clock();
            if(render){
                if (glfwWindowShouldClose(window)) {
                    std::cout << "Finestra chiusa. Terminazione del programma." << std::endl;
                    goto fine;
                }
            }         
            
            int new_n_cell = 0;
            compact_with_thrust(world_id_d, alive_cells_d, world_dim, new_n_cell);
            cudaDeviceSynchronize();        
            *n_cell_alive_h = new_n_cell;


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
                    new_models_weights_d,
                    n_weight,
                    new_models_biases_d,
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
                cudaDeviceSynchronize(); 


            }

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
            
            cudaDeviceSynchronize();        
            

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

            end = clock(); 
            char epocstep[64];
            snprintf(epocstep, sizeof(epocstep), "%d.%d", epoca,step);
            printf("Step: %10s \t alive_cell: %8d  |  %3.1f it/s \n",epocstep,*n_cell_alive_h,1.0f/((float)(end - start) / CLOCKS_PER_SEC));
        }

        // -------------------------------------------
        // FASE 3 : generazione nuove creature 
        // -------------------------------------------

        float *chosen_points = nullptr;
        if(METHOD_EVAL==0) chosen_points=energy_vector_d;
        else chosen_points=occupation_vector_d;

        launch_update_model(
            model_weights_d,
            model_biases_d,
            varation_model_weights_d,
            varation_model_biases_d,
            chosen_points,
            n_weight,
            n_bias,
            n_creature,
            alpha,
            std,
            N_STEPS,
            0
        );    

        // ======================================== Salvataggio nuovo modelli
        if((epoca != 0 && epoca%checkpoint_epoch == 0) || epoca == (N_EPOCH - 1)){    
            cuda_memcpy(model_weights_h,model_weights_d,tot_weights_size,cudaMemcpyDeviceToHost,cc_major,0);        
            cuda_memcpy(model_biases_h,model_biases_d,tot_biases_size,cudaMemcpyDeviceToHost,cc_major,0); 

            if(METHOD_EVAL==0) cuda_memcpy(occupation_vector_h,occupation_vector_d,tot_occupation_vector_size,cudaMemcpyDeviceToHost,cc_major,0);    
            else cuda_memcpy(energy_vector_h,energy_vector_d,tot_energy_vector_size,cudaMemcpyDeviceToHost,cc_major,0); 

            float tot_energy = 0, tot_occupation = 0;
            for(int i=0; i<n_creature; i++){
                if(METHOD_EVAL==0) tot_occupation += occupation_vector_h[i];
                else tot_energy += energy_vector_h[i];
            }

            if(METHOD_EVAL==0) append_score_to_file("points/Occupations_points.txt",tot_occupation/N_STEPS);
            else append_score_to_file("points/Energy_points.txt",tot_energy/N_STEPS); 

            save_model_on_file(path_save_file,model_weights_h,model_biases_h,n_weight,n_bias);
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
    cuda_Free(varation_model_weights_d, cc_major, 0);
    cuda_Free(varation_model_biases_d, cc_major, 0);
    cuda_Free(new_models_weights_d, cc_major, 0);
    cuda_Free(new_models_biases_d, cc_major, 0);
    cuda_Free(workspace_input_d, cc_major, 0);

    free(energy_vector_h);
    free(occupation_vector_h);
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
