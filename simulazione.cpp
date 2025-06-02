
#include "libs/mappa_colori.cuh"
#include "libs/mondo_kernel.cuh"
#include "libs/NN_kernel.cuh"
#include "libs/utils_kernel.cuh"
#include "libs/utils_cpu.h"
#include "libs/perlin_noise.cuh"

#include <GLFW/glfw3.h>

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <unistd.h> 
#include <time.h>
#include <iomanip>



void simulazione(
    int world_dim, int n_creature, 
    int *model_structure, int n_layer, size_t reserve_free_memory, 
    float *weights_models, float *biases_models, 
    int const N_EPHOCS, int const N_STEPS, int const MAX_WORKSPACE, int const METHOD_EVAL, 
    bool render,        // checkpoint_epoc 
    GLFWwindow* window, GLuint textureID
) {


    //  -----------------------------------------
    //
    //  -----------------------------------------
    int checkpoint_epoch = 1;
    
    float PN_scale_obstacles = 10.0f;
    float PN_threshold_obstacles = 0.85f;

    float PN_scale_food = 8.0f;
    float PN_threshold_food = 0.90f;
    float random_threshold_food = 0.99f;
    
    float starting_value = 9.0f;
    float energy_fraction = (1.0f / 9.0f) / 2.0f;
    float energy_decay = 0.001f;

    float winners_fraction = 0.45f;
    float recombination_newborns_fraction = 0.75f;
    
    float gen_x_block = 0.005f;
    float mutation_probability = 0.02f;
    float mutation_range = 0.4f;

    int clean_window_size = 11;

    // parametrizzare food e obstacles, mutazioni randomiche, soglia di ricombinazione

    // sistemare tutti i reset con memsetasync
    // mettere array di supporto all'inizio e si passa come argomento 


    clock_t start;
    clock_t end;

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

    // stream CUDA
    int n_stream = MAX_WORKSPACE;
    int a_stream = -1;
    cudaStream_t streams[n_stream];
    if (cc_major >= 5) {
        for(int i=0;i<n_stream;i++) CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

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
    size_t workspace_size = dim_max_layer  * sizeof(float);

    // compute n_workspace using all free memory avaible
    computeFreeMemory(&free_mem);
    int n_workspace = (free_mem > reserve_free_memory) ? (free_mem - reserve_free_memory) / workspace_size : 0;

    // if are not allocate workspace return error
    if (n_workspace == 0) {
        throw std::runtime_error("No memory for workspace... impossible to continue.");
    } else if (n_workspace > world_dim * world_dim) {
        n_workspace = world_dim * world_dim;
    }
    if(n_workspace>MAX_WORKSPACE) n_workspace=MAX_WORKSPACE;

    std::cout << "Free memory: " << free_mem << " bytes\n";
    std::cout << "Reserved memory: " << reserve_free_memory << " bytes\n";
    std::cout << "Workspace size per slot: " << workspace_size << " bytes\n";
    std::cout << "Allocate " << n_workspace << " workspace slots for the simulation.\n";

    // allocazione workspace
    float *workspace_input_d  = (float*) cuda_allocate(n_workspace * workspace_size, cc_major, 0);
    // float *workspace_output_d = (float*) cuda_allocate(n_workspace * dim_workspace, cc_major, 0);

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

        /*
        (per debuggare)        
        cudaMemset(model_weights_d, 0, n_weight*5*sizeof(float));    
        CUDA_CHECK(cudaGetLastError());          
        
        cudaMemset(model_biases_d,0, n_bias*5*sizeof(float));    
        CUDA_CHECK(cudaGetLastError());      
        */
        
        // Load on CPU vettore pesi tutti i modelli (world_weights_h è vettore host con dati)
        cuda_memcpy(model_weights_h, model_weights_d, tot_models_weight_size, cudaMemcpyDeviceToHost, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        // Load on CPU vettore bias tutti i modelli (world_biases_h è vettore host con dati)
        cuda_memcpy(model_biases_h, model_biases_d, tot_models_bias_size, cudaMemcpyDeviceToHost, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        
        save_model_on_file("models/file1.txt",model_structure,n_layer,model_weights_h,model_biases_h,n_weight,n_bias,n_creature);

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

    for (int epoca = 0; epoca < N_EPHOCS; epoca++) {
        std::cout << "=======================  Epoca: " << epoca << "  ========================\n";

        // -------------------------------------------
        // FASE 1 : preparazione epoca 
        // -------------------------------------------
        

        /*
        // - Azzeramento mondo valori,id,signaling (da eliminare), usare memset(arr, 0, sizeof(arr)); la copia in gpu avviene dopo, sovrascrivendo l'array vecchio
        launch_reset_kernel<float>(world_value_d, world_dim * world_dim, 0);
        CUDA_CHECK(cudaGetLastError());
        launch_reset_kernel<int>(world_id_d, world_dim * world_dim, 0);
        CUDA_CHECK(cudaGetLastError());
        // - Passo il mondo valori ed ID sulla CPU post RESET
        // - Possibile ottimizzazione?
        cuda_memcpy(world_value_h, world_value_d, tot_world_dim_size_float, cudaMemcpyDeviceToHost, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        cuda_memcpy(world_id_h, world_id_d, tot_world_dim_size_int, cudaMemcpyDeviceToHost, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        launch_reset_kernel<float>(world_signal_d, world_dim * world_dim, 0);
        CUDA_CHECK(cudaGetLastError());
 
        // - Azzeramento vettore valutazione x occupazione ed energia
        launch_reset_kernel<float>(energy_vector_d, n_creature, 0);
        CUDA_CHECK(cudaGetLastError());
        launch_reset_kernel<float>(occupation_vector_d, n_creature, 0);
        CUDA_CHECK(cudaGetLastError());
        */

        std::memset(world_value_h, 0, tot_world_dim_size_float);
        std::memset(world_id_h, 0, tot_world_dim_size_int);
        std::memset(alive_cells_h, 0, tot_world_dim_size_int);

        cudaMemset(world_signal_d, 0, tot_world_dim_size_float);    
        CUDA_CHECK(cudaGetLastError());    
        cudaMemset(energy_vector_d, 0, tot_energy_vector_size);
        CUDA_CHECK(cudaGetLastError());
        cudaMemset(occupation_vector_d, 0, tot_occupation_vector_size);
        CUDA_CHECK(cudaGetLastError());

        printf("RESET ALL MATRIX \n"); 

        // - Aggiunta creature al mondo 
        *n_cell_alive_h = 0;

        int random_index = rand() % world_dim*world_dim;
        for (int i = 0; i < n_creature; i++){
            while(world_id_h[random_index] != 0){
                random_index = rand() % (world_dim*world_dim);
            }
            if(world_id_h[random_index] == 0){
                world_value_h[random_index] = starting_value;
                world_id_h[random_index] = i + 1;
                alive_cells_h[i] = random_index;
                *n_cell_alive_h += 1;
                random_index = rand() % (world_dim*world_dim);
            }     
        }

        printf("SETUP CELL ALIVE \n");

        // - Passo il mondo valori ed ID sulla GPU
        cuda_memcpy(world_value_d, world_value_h, tot_world_dim_size_float, cudaMemcpyHostToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());

        cuda_memcpy(world_id_d, world_id_h, tot_world_dim_size_int, cudaMemcpyHostToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());

        // - Passo il vettore cellule vive dalla CPU a GPU
        cuda_memcpy(alive_cells_d, alive_cells_h, tot_world_dim_size_int, cudaMemcpyHostToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());

        // - Aggiunta ostacoli perlin al mondo
        launch_perlinNoise_obstacles(world_dim, world_id_d, PN_scale_obstacles, PN_threshold_obstacles, 0);
        CUDA_CHECK(cudaGetLastError());
        
        // - Aggiunta cibo perlin al mondo
        launch_perlinNoise_food(world_dim, world_id_d, world_value_d, PN_scale_food, PN_threshold_food, 0);
        CUDA_CHECK(cudaGetLastError());

        // - Aggiunta cibo al mondo
        // - Possibile ottimizzazione togliendo i curandstate
        launch_add_objects_to_world(world_value_d, world_id_d, world_dim, 0, 0.3f, 1.0f, random_threshold_food, 0);
        CUDA_CHECK(cudaGetLastError());

        //crea una zona di vuoto attorno alle cellele appena nate (diminuisce l'influenza della fortuna nello spawn)
        launch_clean_around_cells(world_value_d, world_id_d, world_dim, alive_cells_d, n_cell_alive_h, clean_window_size, 0);
        CUDA_CHECK(cudaGetLastError());
        /*
        // - Ritorno mondo valori e mondo id definitivi su CPU per debug         
        cuda_memcpy(world_value_h, world_value_d, tot_world_dim_size_float, cudaMemcpyDeviceToHost, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        cuda_memcpy(world_id_h, world_id_d, tot_world_dim_size_int, cudaMemcpyDeviceToHost, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        */

        cudaDeviceSynchronize();

        /*======================================================================================================================================*/
        for(int step=0; step<N_STEPS && *n_cell_alive_h > 0; step++){
            start = clock();
            //printf("CELLULE VIVE = %d \n",*n_cell_alive_h);
            // attivazione del render se il flag è attivo
            if(render){
                if (glfwWindowShouldClose(window)) {
                    std::cout << "Finestra chiusa. Terminazione del programma." << std::endl;
                    goto fine;
                }
            }            

            // -------------------------------------------
            // FASE 2 : calcolo step 
            // -------------------------------------------
            int offset=0;
            int vision = sqrt(dim_input/2);

            cudaMemset(world_contributions_d, 0, tot_matrix_contribution_size);
            CUDA_CHECK(cudaGetLastError());
            cudaMemset(workspace_input_d, 0, workspace_size*dim_max_layer);
            CUDA_CHECK(cudaGetLastError());

            /*
            // reset matrice dei contributi e workspace 
            launch_reset_kernel<float>(world_contributions_d, world_dim * world_dim * n_creature, 0);
            CUDA_CHECK(cudaGetLastError());
            launch_reset_kernel<float>(workspace_input_d, n_workspace*dim_max_layer, 0);
            CUDA_CHECK(cudaGetLastError());
            */

            //inizializzo l'offset per le cellule per trovare la corrispettiva stazione di lavoro
            int offset_alive_cell = 0;
            while(offset_alive_cell<*n_cell_alive_h){

                int max = n_workspace<*n_cell_alive_h?n_workspace:*n_cell_alive_h;

                for(int workspace_idx=0; workspace_idx<max && offset_alive_cell<*n_cell_alive_h; workspace_idx++){

                    int offset_workspace_in = dim_max_layer*workspace_idx;
                    // int offset_workspace_out = output_size*workspace_idx;
                    int stream_id = workspace_idx % n_stream;

                    launch_vision(
                        world_value_d,
                        world_id_d,
                        world_signal_d,
                        world_dim,
                        alive_cells_d+offset_alive_cell,
                        vision,
                        workspace_input_d+offset_workspace_in,
                        streams[stream_id]
                    );
                    CUDA_CHECK(cudaGetLastError());


                    launch_NN_forward(
                        workspace_input_d+offset_workspace_in,
                        workspace_input_d+offset_workspace_in,
                        model_weights_d,
                        n_weight,
                        model_biases_d,
                        n_bias,
                        model_structure,
                        offset_alive_cell,
                        alive_cells_d,
                        world_id_d,
                        n_layer,
                        streams[stream_id]
                    );
                    CUDA_CHECK(cudaGetLastError());

                    launch_output_elaboration(
                        world_value_d,
                        world_signal_d,
                        world_id_d,
                        world_contributions_d,
                        workspace_input_d+offset_workspace_in,
                        alive_cells_d,
                        world_dim,
                        n_creature,
                        dim_output,
                        offset_alive_cell,
                        energy_fraction,
                        streams[stream_id]
                    );
                    CUDA_CHECK(cudaGetLastError());

                    offset_alive_cell++;
                    
                }

                // Aspetto che tutti i kernel del batch finiscano prima di riutilizzare i workspace
                /*
                for(int workspace_idx = 0; workspace_idx < max; workspace_idx++) {
                    int stream_id = workspace_idx % n_stream;
                    cudaStreamSynchronize(streams[stream_id]);
                }
                */

            }
            
            launch_world_update(
                world_value_d,
                world_id_d,
                world_signal_d,
                world_contributions_d,
                alive_cells_d,
                world_dim,
                n_creature,
                n_cell_alive_d,
                energy_decay,
                0
            );
            CUDA_CHECK(cudaGetLastError());


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

            // se il render è attivo genero la schermata con openGL
            if(render){

                //launch_mappa_colori(world_value_d, world_id_d, world_rgb_d, world_dim, 0);
                //CUDA_CHECK(cudaGetLastError());

                launch_mappa_signal(world_value_d, world_id_d, world_signal_d, world_rgb_d, world_dim, n_creature, 0);
                CUDA_CHECK(cudaGetLastError());

                cudaMemcpy(world_rgb_h, world_rgb_d, tot_world_dim_size_float * 3, cudaMemcpyDeviceToHost);
                CUDA_CHECK(cudaGetLastError());

                // Carica i dati nella texture OpenGL
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, world_dim, world_dim, GL_RGB, GL_FLOAT, world_rgb_h);

                //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, world_dim, world_dim, 0, GL_RGB, GL_FLOAT, world_rgb_h);

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

            // - Aggiornamento vettori valutazione occupazione ed energia
            launch_compute_energy_and_occupation(world_value_d,world_id_d,occupation_vector_d,energy_vector_d,world_dim,n_creature, 0);
            CUDA_CHECK(cudaGetLastError());

            /*
            // - ritorno i valori per debug di fatto non serve siccome gia lavoro sulla GPU
            // - Ritorno mondo valori
            cuda_memcpy(world_value_h, world_value_d, tot_world_dim_size_float, cudaMemcpyDeviceToHost, cc_major, streams[0]);
            CUDA_CHECK(cudaGetLastError());
            // - Ritorno mondo id 
            cuda_memcpy(world_id_h, world_id_d, tot_world_dim_size_int, cudaMemcpyDeviceToHost, cc_major, streams[0]);
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
            end = clock();  // End time
            std::cout << "Step: " << epoca << "." << step << " " << *n_cell_alive_h << "  |" << std::fixed << std::setprecision(1) << 1.0f/((float)(end - start) / CLOCKS_PER_SEC) << " it/s" << "\n";
        }

        // -------------------------------------------
        // FASE 3 : generazione nuove creature 
        // -------------------------------------------

        seed = (unsigned long)time(NULL);

        // imposto una percentuale di quante creature voglio tenere per la rimescolazione genetica
        int limit_winner = n_creature * winners_fraction;
        int limit_recombination = n_creature * recombination_newborns_fraction;

        // - spostamento vettore energia su HOST
        cuda_memcpy(energy_vector_h,energy_vector_d,tot_energy_vector_size,cudaMemcpyDeviceToHost,cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        // - spostamento vettore occupazione su HOST
        cuda_memcpy(occupation_vector_h,occupation_vector_d,tot_occupation_vector_size,cudaMemcpyDeviceToHost,cc_major, 0);
        CUDA_CHECK(cudaGetLastError());

        float tot_score = 0;

        // - Ordinamento punteggi creature
        if(METHOD_EVAL==0){
            tot_score = argsort_bubble(energy_vector_h,creature_ordered_h,n_creature);
        } 
        if(METHOD_EVAL==1){
            tot_score = argsort_bubble(occupation_vector_h,creature_ordered_h,n_creature); 
        } 
        
        tot_score = tot_score / n_creature;

        append_score_to_file("log_score.txt", tot_score);
        /*
        printf("-----------------------------PUNTEGGI CREATURE----------------------\n");
        for(int i=0; i<n_creature; i++) printf("%3d) energy: %.6f occupation: %.6f\n",i+1,energy_vector_h[i],occupation_vector_h[i]);
        printf("-----------------------------END PUNTEGGI CREATURE----------------------\n");
        printf("CREATURE ORDER, LIMIT_winner(%d): \n",limit_winner);
        for(int i=0; i<n_creature; i++) printf("%3d ",creature_ordered_h[i]+1);
        printf("\n");
        
        */
        int first_ID = creature_ordered_h[0];
        if (first_ID < 0 || first_ID >= n_creature) {
            std::cerr << "Errore: first_ID fuori range: " << first_ID << "\n";
        }
        //il primo vince sempre
        cuda_memcpy(new_model_weights_d, model_weights_d + n_weight * first_ID, n_weight * sizeof(float), cudaMemcpyDeviceToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        
        cuda_memcpy(new_model_biases_d, model_biases_d + n_bias * first_ID, n_bias * sizeof(float), cudaMemcpyDeviceToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());  

    

        // - Creazione nuove creature 
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
        
        // Generation random models
        launch_fill_random_kernel(new_model_weights_d,n_weight*limit_recombination,n_weight*n_creature,-1.0f,1.0f,seed + 1, 0);
        CUDA_CHECK(cudaGetLastError());
        launch_fill_random_kernel(new_model_biases_d,n_bias*limit_recombination,n_bias*n_creature,-1.0f,1.0f,seed + 2, 0);
        CUDA_CHECK(cudaGetLastError());
        


        cudaDeviceSynchronize();


        // - Load vettore nuovi pesi su vettore vecchi pesi
        cuda_memcpy(model_weights_d, new_model_weights_d, tot_models_weight_size, cudaMemcpyDeviceToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());
        // - Load vettore nuovi bias su vettore vecchi bias 
        cuda_memcpy(model_biases_d, new_model_biases_d, tot_models_bias_size, cudaMemcpyDeviceToDevice, cc_major, 0);
        CUDA_CHECK(cudaGetLastError());

        if((epoca != 0 && epoca%checkpoint_epoch == 0) || epoca == (N_EPHOCS - 1)){
            // - Ritorno vettore nuovi pesi creature
            cuda_memcpy(model_weights_h, new_model_weights_d, tot_models_weight_size, cudaMemcpyDeviceToHost, cc_major, 0);
            CUDA_CHECK(cudaGetLastError());
            // - Ritorno vettore nuovi bias creature 
            cuda_memcpy(model_biases_h, new_model_biases_d, tot_models_bias_size, cudaMemcpyDeviceToHost, cc_major, 0);
            CUDA_CHECK(cudaGetLastError());

            save_model_on_file("models/file1.txt",model_structure,n_layer,model_weights_h,model_biases_h,n_weight,n_bias,n_creature);
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

    // Free zone di memoria GPU
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
    // cuda_Free(n_cell_alive_d,cc_major, streams[0]);
    // cuda_Free(creature_ordered_d,cc_major,streams[0]);
    // cuda_Free(workspace_output_d, cc_major, streams[0]);


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
    

    for(int i=0; i<n_stream; i++) cudaStreamDestroy(streams[i]);

    std::cout << "End Simulation. \n";
}
