#include "libs/mondo_kernel.cu"
#include "libs/NN_kernel.cu"
#include "libs/utils_kernel.cu"
#include "libs/utils_cpu.cu"

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>


void simulazione(
    int world_dim, int n_creature, 
    int *dims_model, int n_layer, size_t reserve_free_memory, 
    float *weights_models, float *biases_models, 
    int const N_EPHOCS, int const N_STEPS, int const MAX_WORKSPACE, int const METHOD_EVAL
) {


    FILE *file = fopen("output.txt","w");
    fprintf(file,"%d\n",world_dim);

    int input_size = dims_model[0];
    int output_size = dims_model[n_layer-1];


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
        n_weight += dims_model[i] * dims_model[i + 1];
        n_bias += dims_model[i + 1];
    }

    size_t tot_world_dim_size = sizeof(float) * world_dim * world_dim;
    size_t tot_matrix_contribution_size = tot_world_dim_size * n_creature;
    size_t tot_eval_vector_size = n_creature * sizeof(float);
    size_t tot_models_weight_size = n_creature * n_weight * sizeof(float);
    size_t tot_models_bias_size = n_creature * n_bias * sizeof(float);
    size_t tot_support_vector_size = tot_world_dim_size;

    // Allocazioni CPU
    float *world_value_h       = (float*) malloc(tot_world_dim_size);
    int   *world_id_h          = (int*)   malloc(tot_world_dim_size);
    float *energy_vector_h     = (float*) malloc(tot_eval_vector_size);
    int   *occupation_vector_h = (int*)   malloc(tot_eval_vector_size);
    int   *creature_ordered_h  = (int*)   malloc(tot_eval_vector_size);
    int   *n_cell_alive_h      = (int*)   malloc(sizeof(int));
    cudaMallocManaged(&n_cell_alive_h, sizeof(int));
    float *model_weights_h     = nullptr;
    float *model_biases_h      = nullptr;

    if(weights_models==nullptr) model_weights_h = (float*) malloc(tot_models_weight_size);
    else model_weights_h       = weights_models;
    if(biases_models==nullptr)  model_biases_h = (float*) malloc(tot_models_bias_size);
    else model_biases_h        = biases_models;

    // Creazione Mondo con creature ------------------------------------------------------Cambiato
    for (int i = 0; i < n_creature; i++){
        int random_index = rand() % world_dim*world_dim;
        world_value_h[random_index] = 1;
        world_id_h[random_index] = i + 1;
    }

    // stream CUDA
    int n_stream = MAX_WORKSPACE;
    int a_stream = -1;
    cudaStream_t streams[n_stream];
    if (cc_major >= 5) {
        for(int i=0;i<n_stream;i++) CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocazioni GPU con stream diversi
    
    float *world_RGB           = (float*) cuda_allocate(tot_world_dim_size*3, cc_major, streams[0]);
    float *world_value_d       = (float*) cuda_allocate(tot_world_dim_size, cc_major, streams[0]);
    int   *world_id_d          = (int*) cuda_allocate(tot_matrix_contribution_size, cc_major, streams[0]);
    float *model_weights_d     = (float*) cuda_allocate(tot_models_weight_size, cc_major, streams[0]);
    float *model_biases_d      = (float*) cuda_allocate(tot_models_bias_size, cc_major, streams[0]);
    int   *alive_cells_d       = (int*)   cuda_allocate(tot_world_dim_size, cc_major, streams[0]);
    float *energy_vector_d     = (float*) cuda_allocate(tot_eval_vector_size, cc_major, streams[0]);
    int   *occupation_vector_d = (int*)   cuda_allocate(tot_eval_vector_size, cc_major, streams[0]);
    //int   *creature_ordered_d  = (int*)   cuda_allocate(tot_eval_vector_size, cc_major, streams[0]);
    float *new_model_weights_d = (float*) cuda_allocate(tot_models_weight_size, cc_major, streams[0]);
    float *new_model_biases_d  = (float*) cuda_allocate(tot_models_bias_size, cc_major, streams[0]);
    int   *n_cell_alive_d      = (int*)   cuda_allocate(sizeof(int), cc_major, streams[0]);
    int   *support_vector_d    = (int*)   cuda_allocate(tot_support_vector_size, cc_major, streams[0]); // zona di memoria allocata per operazioni di supporto
    int   *NN_structure_d      = (int*)   cuda_allocate(n_layer * sizeof(int), cc_major, streams[0]);

    cuda_memcpy(NN_structure_d, dims_model, n_layer * sizeof(int), cudaMemcpyHostToDevice, cc_major, streams[0]);
    CUDA_CHECK(cudaGetLastError());
    //cudaDeviceSynchronize();

    // Calcolo spazio massimo disponibile per workspace
    int dim_input = dims_model[0];
    int dim_output = dims_model[n_layer - 1];
    size_t dim_workspace = (dim_input + dim_output) * sizeof(float);
    computeFreeMemory(&free_mem);

    size_t n_workspace = (free_mem > reserve_free_memory) ? (free_mem - reserve_free_memory) / dim_workspace : 0;

    if (n_workspace == 0) {
        throw std::runtime_error("No memory for workspace... impossible to continue.");
    } else if (n_workspace > world_dim * world_dim) {
        n_workspace = world_dim * world_dim;
    }
    if(n_workspace>MAX_WORKSPACE) n_workspace=MAX_WORKSPACE;

    std::cout << "Free memory: " << free_mem << " bytes\n";
    std::cout << "Reserved memory: " << reserve_free_memory << " bytes\n";
    std::cout << "Workspace size per slot: " << dim_workspace << " bytes\n";
    std::cout << "Allocate " << n_workspace << " workspace slots for the simulation.\n";

    // allocazione workspace
    float *workspace_input_d  = (float*) cuda_allocate(n_workspace * dim_input  * sizeof(float), cc_major, streams[0]);
    float *workspace_output_d = (float*) cuda_allocate(n_workspace * dim_output * sizeof(float), cc_major, streams[0]);

    computeFreeMemory(&free_mem);
    std::cout << "Free memory after allocation " << free_mem/1024 << " KB\n";

    printf("END ALLOCATIONS\n");

    // CARIMENTO DATI
    if(biases_models==nullptr && weights_models==nullptr){
        // Generation random models
        launch_fill_random_kernel(model_weights_d,n_weight*n_creature,-1.0f,1.0f,0,streams[0]);
        CUDA_CHECK(cudaGetLastError());
        launch_fill_random_kernel(model_biases_d,n_bias*n_creature,-1.0f,1.0f,0,streams[0]);
        CUDA_CHECK(cudaGetLastError());

        // Load on CPU vettore pesi tutti i modelli (world_weights_h è vettore host con dati)
        cuda_memcpy(model_weights_h, model_weights_d, tot_models_weight_size, cudaMemcpyDeviceToHost, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        // Load on GPU vettore bias tutti i modelli (world_biases_h è vettore host con dati)
        cuda_memcpy(model_biases_h, model_biases_d, tot_models_bias_size, cudaMemcpyDeviceToHost, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());

        save_model_on_file("models/file1.txt",dims_model,n_layer,model_weights_h,model_biases_h,n_weight,n_bias,n_creature);

        printf("FINE GENERAZIONE MODELLI \n");
    }

    for (int epoca = 0; epoca < N_EPHOCS; epoca++) {
        std::cout << "Epoca " << epoca << "\n";

        // -------------------------------------------
        // FASE 1 : preparazione epoca 
        // -------------------------------------------

        // Load on GPU vettore pesi tutti i modelli (world_weights_h è vettore host con dati)
        cuda_memcpy(model_weights_d, model_weights_h, tot_models_weight_size, cudaMemcpyHostToDevice, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        // Load on GPU vettore bias tutti i modelli (world_biases_h è vettore host con dati)
        cuda_memcpy(model_biases_d, model_biases_h, tot_models_bias_size, cudaMemcpyHostToDevice, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());

        printf("LOAD MODEL ON GPU\n");

        // - Azzeramento mondo valori,id,contributi,signaling 
        launch_reset_kernel<float>(world_value_d, world_dim * world_dim, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        launch_reset_kernel<int>(world_id_d, world_dim * world_dim, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        launch_reset_kernel<float>(contribution_d, world_dim * world_dim * n_creature, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        launch_reset_kernel<float>(signaling_d, world_dim * world_dim, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        // - Azzeramento vettore valutazione x occupazione ed energia
        launch_reset_kernel<float>(energy_vector_d, n_creature, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        launch_reset_kernel<int>(occupation_vector_d, n_creature, streams[0]);
        CUDA_CHECK(cudaGetLastError());

        printf("RESET ALL MATRIX \n");
        
        // - Aggiunta creature al mondo -------------------------------------------------------Cambiato
        cuda_memcpy(world_value_d, world_value_h, tot_world_dim_size, cudaMemcpyHostToDevice, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());

    

        
        // - Aggiunta ostacoli al mondo
        launch_add_objects_to_world(world_value_d, world_id_d, world_dim, -1, 1.0f, 1.0f, 0.9f, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        /**/


        // - Aggiunta cibo al mondo
        launch_add_objects_to_world(world_value_d, world_id_d, world_dim, 0, 0.3f, 1.0f, 0.2f, streams[0]);
        CUDA_CHECK(cudaGetLastError());

        printf("ADD ELEMENTS TO WORLD\n");
            
        /*
        // - Calcolo cellule vive
        launch_find_index_cell_alive(world_id_d,world_dim*world_dim,alive_cells_d,n_cell_alive_d,n_cell_alive_h,support_vector_d,streams[0]);
        CUDA_CHECK(cudaGetLastError());
        
        printf("COMPUTE ALIVE CELLS %d\n",*n_cell_alive_h);
        */

        // - Ritorno mondo valori
        cuda_memcpy(world_value_h, world_value_d, tot_world_dim_size, cudaMemcpyDeviceToHost, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        // - Ritorno mondo id 
        cuda_memcpy(world_id_h, world_id_d, tot_world_dim_size, cudaMemcpyDeviceToHost, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());

        save_map(file,world_dim,world_value_h,world_id_h);

        printf("RETURN VIEW WORLD SETUP\n");

        for(int step=0; step<N_STEPS; step++){
            std::cout << "Step " << step << "\n";

            // -------------------------------------------
            // FASE 2 : calcolo step 
            // -------------------------------------------
            int offset=0;
            int vision = sqrt(dim_input/3);

            launch_reset_kernel<float>(contribution_d, world_dim * world_dim * n_creature, streams[0]);
            CUDA_CHECK(cudaGetLastError());

            while(offset<*n_cell_alive_h){

                for(int i=0; i<n_workspace; i++){
                    int idx_cell = i+offset;
                    if(idx_cell<*n_cell_alive_h){

                        // - Calcolo input 
                        launch_vision(world_value_d,world_id_d,signaling_d,world_dim,alive_cells_d+idx_cell,vision,i,workspace_input_d,streams[i]);
                        CUDA_CHECK(cudaGetLastError());
                        printf("LANCIO KERNEL VISION \n");

                        NN_forward(
                            workspace_input_d,
                            workspace_output_d,
                            model_weights_d,
                            n_weights,
                            model_biases_d,
                            n_biases,
                            NN_structure_d,
                            idx_cell,
                            alive_cells_d,
                            world_id_d    
                        );

                        CUDA_CHECK(cudaGetLastError());
                        printf("NN_forward \n");

                    }
                    cudaDeviceSynchronize();
                    

                }
                output_elaboration(
                    signaling_d,
                    contribution_d,
                    world_id_d,
                    n_creature,
                    world_dim,
                    workspace_output_d,
                    output_size,
                    alive_cells_d,
                    offset,
                    n_workspace
                );
                CUDA_CHECK(cudaGetLastError());
                printf("CELL %2d & %2d END WORK \n",offset,offset+n_workspace);
                offset+=n_workspace;


            }
                
            launch_world_update(contribution_d,
                                world_value_d,
                                world_id_d,
                                world_dim,
                                n_creature,
                                n_cell_alive_h, 
                                step);
                                    

            CUDA_CHECK(cudaGetLastError());



            launch_cellule_cleanup(alive_cells_d, 
                                    n_cell_alive_h, 
                                    world_id_d);

            CUDA_CHECK(cudaGetLastError());

            
           

            // - Aggiornamento vettori valutazione occupazione ed energia
            launch_compute_energy_and_occupation(world_value_d,world_id_d,occupation_vector_d,energy_vector_d,world_dim,n_creature,streams[0]);
            CUDA_CHECK(cudaGetLastError());

            wrap_mappaColori(mondo, id_matrix, mondo_rgb, dim_mondo, dim_mondo);
            printf("UPDATE EVALUATION VECTORS \n");

            // - Reset matrice dei contributi
            
            CUDA_CHECK(cudaGetLastError());
            printf("RESET CONTRIBUTION MATRIX \n");
            // - Ritorno mondo valori
            cuda_memcpy(world_value_h, world_value_d, tot_world_dim_size, cudaMemcpyDeviceToHost, cc_major, streams[0]);
            CUDA_CHECK(cudaGetLastError());
            // - Ritorno mondo id 
            cuda_memcpy(world_id_h, world_id_d, tot_world_dim_size, cudaMemcpyDeviceToHost, cc_major, streams[0]);
            CUDA_CHECK(cudaGetLastError());

            if(render){
                if (glfwWindowShouldClose(window)) {
                    std::cout << "Finestra chiusa. Terminazione del programma." << std::endl;
                    break; // Esce dal ciclo
                }
            }

            

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


            save_map(file,world_dim,world_value_h,world_id_h);

            printf("RETURN VIEW WORLD %d.%d \n",epoca,step);

        }

        // -------------------------------------------
        // FASE 3 : generazione nuove creature 
        // -------------------------------------------

        int limit = n_creature * 0.4f;

        // - spostamento vettore energia su HOST
        cuda_memcpy(energy_vector_h,energy_vector_d,tot_eval_vector_size,cudaMemcpyDeviceToHost,cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        // - spostamento vettore occupazione su HOST
        cuda_memcpy(occupation_vector_h,occupation_vector_d,tot_eval_vector_size,cudaMemcpyDeviceToHost,cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());

        // - Ordinamento punteggi creature
        if(METHOD_EVAL==0) argsort_bubble(energy_vector_h,creature_ordered_h,n_creature);
        if(METHOD_EVAL==1) argsort_bubble(occupation_vector_h,creature_ordered_h,n_creature);

        printf("-----------------------------PUNTEGGI CREATURE----------------------\n");
        for(int i=0; i<n_creature; i++) printf("%3d) energy: %f occupation: %d\n",i+1,energy_vector_h[i],occupation_vector_h[i]);
        printf("-----------------------------END PUNTEGGI CREATURE----------------------\n");
        printf("CREATURE ORDER, LIMIT(%d): \n",limit);
        for(int i=0; i<n_creature; i++) printf("%3d ",creature_ordered_h[i]+1);
        printf("\n");

        // - Creazione nuove creature 
        for(int i=0;i<n_creature;i++){

            int idx1 = get_random_int(0,limit);
            int idx2 = get_random_int(0,limit);
            int gen1 = creature_ordered_h[idx1];
            int gen2 = creature_ordered_h[idx2];

            launch_recombine_models_kernel(model_weights_d,model_biases_d,
                new_model_weights_d,new_model_biases_d,
                n_weight, n_bias, gen1, gen2,
                i, 0.2f, 0.30f, 1.0f, 0, streams[next_stream(&a_stream,n_stream)]
            );
            CUDA_CHECK(cudaGetLastError());

        }


        // - Ritorno vettore nuovi pesi creature
        cuda_memcpy(model_weights_h, new_model_weights_d, tot_models_weight_size, cudaMemcpyDeviceToHost, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        // - Ritorno vettore nuovi bias creature 
        cuda_memcpy(model_biases_h, new_model_biases_d, tot_models_bias_size, cudaMemcpyDeviceToHost, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());

        // - Load vettore nuovi pesi su vettore vecchi pesi
        cuda_memcpy(model_weights_d, new_model_weights_d, tot_models_weight_size, cudaMemcpyDeviceToDevice, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());
        // - Load vettore nuovi bias su vettore vecchi bias 
        cuda_memcpy(model_biases_d, new_model_biases_d, tot_models_bias_size, cudaMemcpyDeviceToDevice, cc_major, streams[0]);
        CUDA_CHECK(cudaGetLastError());

        save_model_on_file("models/file1.txt",dims_model,n_layer,model_weights_h,model_biases_h,n_weight,n_bias,n_creature);
        printf("MODEL GENERATE AND SAVE \n");

    }


    // -------------------------------------------
    // POST-FASE : 
    // -------------------------------------------


    // Free zone di memoria GPU
    cuda_Free(world_value_d, cc_major, streams[0]);
    cuda_Free(world_id_d, cc_major, streams[0]);
    cuda_Free(contribution_d, cc_major, streams[0]);
    cuda_Free(signaling_d, cc_major, streams[0]);
    cuda_Free(model_weights_d, cc_major, streams[0]);
    cuda_Free(model_biases_d, cc_major, streams[0]);
    cuda_Free(alive_cells_d, cc_major, streams[0]);
    cuda_Free(occupation_vector_d, cc_major, streams[0]);
    cuda_Free(energy_vector_d, cc_major, streams[0]);
    //cuda_Free(creature_ordered_d,cc_major,streams[0]);
    cuda_Free(workspace_input_d, cc_major, streams[0]);
    cuda_Free(workspace_output_d, cc_major, streams[0]);
    cuda_Free(new_model_weights_d, cc_major, streams[0]);
    cuda_Free(new_model_biases_d, cc_major, streams[0]);
    cuda_Free(n_cell_alive_d,cc_major, streams[0]);

    free(world_value_h);
    free(world_id_h);
    free(energy_vector_h);
    free(occupation_vector_h);
    free(creature_ordered_h);
    free(model_weights_h);
    free(model_biases_h);
    free(n_cell_alive_h);
    
    
    printf("FREE DONE\n");
    

    for(int i=0; i<n_stream; i++) cudaStreamDestroy(streams[i]);

    fclose(file);

    std::cout << "End Simulation. \n";
}

int main() {
    int const n_layer = 3;
    int v[n_layer] = {4,5,2};
    size_t dim_free = 1024*1024;
    int dim_world = 10;
    int n_creature = 10;
    int const EPHOCS = 10;
    int const STEPS = 400;
    int const MAX_WORKSPACE = 10;
    int const EVAL_TYPE = 1;
    float *weight_model = nullptr;
    float *bias_model = nullptr;
    simulazione(dim_world,n_creature,v,n_layer,dim_free,weight_model,bias_model,EPHOCS,STEPS,MAX_WORKSPACE,EVAL_TYPE);
    return 0;
}
