#include "libs/mondo_kernel.cu"
#include "libs/NN_kernel.cu"
#include "libs/utils_kernel.cu"
#include "libs/utils_cpu.cu"

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

float random_float(float min, float max) {
    return min + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (max - min);
}

int calcola_weights(int *v, int n){
    int sum = 0;
    for(int i=0; i<n-1; i ++) {
        sum += (v[i]*v[i+1]);
    }
    return sum;
}

int calcola_biases(int *v, int n){
    int sum = 0;
    for(int i=1; i<n; i ++) {
        sum += (v[i]);
    }
    return sum;
}

int main() {

    int const n          = 3;
    int world_dim        = 10;
    int numero_workspace = 10;
    int visione          = 5;
    int input_dim        = visione*visione*3;
    int output_dim       = (3*3)+1;
    int alive_cell       = 21;
    int structure[n]     = {input_dim, 20, output_dim};
    int n_weights        = calcola_weights(structure,n);
    int n_baias          = calcola_biases(structure,n);
    int n_modelli        = 1;

    cudaStream_t stream = 0;

    size_t size_world               = world_dim*world_dim*sizeof(int);
    size_t size_inputs              = numero_workspace*input_dim*sizeof(float);
    size_t size_outputs             = numero_workspace*output_dim*sizeof(float);
    size_t size_weight              = n_weights*n_modelli*sizeof(float);
    size_t size_bias                = n_baias*n_modelli*sizeof(float);
    size_t size_contribution_matrix = size_world*n_modelli;

    float *world_value_h                = (float*)  malloc(size_world);
    float *world_signal_h               = (float*)  malloc(size_world);
    int   *world_id_h                   = (int*)    malloc(size_world);
    float *inputs_h                     = (float*)  malloc(size_inputs);
    float *outputs_h                    = (float*)  malloc(size_outputs);
    float *weight_h                     = (float*)  malloc(size_weight);
    float *baias_h                      = (float*)  malloc(size_bias);
    int   *alive_cell_h                 = (int*)    malloc(size_world);
    float *contribution_matrix_h        = (float*)  malloc(size_contribution_matrix);  

    float *world_value_d;   
    float *world_signal_d;   
    int   *world_id_d;         
    int   *alive_cell_d;       
    float *inputs_d;
    float *outputs_d;   
    float *weight_d;
    float *baias_d;     
    float *contribution_matrix_d;

    cudaMalloc((void**) &world_value_d, size_world);
    cudaMalloc((void**) &world_signal_d, size_world);
    cudaMalloc((void**) &world_id_d, size_world);
    cudaMalloc((void**) &alive_cell_d, size_world);
    cudaMalloc((void**) &inputs_d, size_inputs);
    cudaMalloc((void**) &outputs_d, size_outputs);
    cudaMalloc((void**) &weight_d, size_weight);
    cudaMalloc((void**) &baias_d, size_bias);
    cudaMalloc((void**) &contribution_matrix_d, size_contribution_matrix);

    for(int i=0; i<world_dim*world_dim; i++) {
        world_value_h[i] = random_float(0,1);
        world_id_h[i] = random_float(-1-1,n_modelli+1);
        world_signal_h[i] = random_float(0,10);
        alive_cell_h[i] = random_float(0,world_dim*world_dim);
    }

    for(int i=0; i<alive_cell; i++){
        world_id_h[alive_cell_h[i]] = random_float(1,n_modelli);
    }

    for(int i=0; i<n_modelli*world_dim; i++){
        contribution_matrix_h[i] = 0;
    }

    for(int i=0; i<n_weights; i++){
        weight_h[i] = random_float(-1,1);
    }

    for(int i=0; i<n_baias; i++){
        baias_h[i] = random_float(-1,1);
    }

    cudaMemcpy(world_value_d,           world_value_h,  size_world,                 cudaMemcpyHostToDevice);
    cudaMemcpy(world_signal_d,          world_signal_h, size_world,                 cudaMemcpyHostToDevice);
    cudaMemcpy(world_id_d,              world_id_h,     size_world,                 cudaMemcpyHostToDevice);
    cudaMemcpy(alive_cell_d,            alive_cell_h,   size_world,                 cudaMemcpyHostToDevice);
    cudaMemset(inputs_d,                0,              size_inputs);
    cudaMemset(outputs_d,               0,              size_outputs);
    cudaMemcpy(weight_d,                weight_h,       size_weight,                cudaMemcpyHostToDevice);
    cudaMemcpy(baias_d,                 baias_h,        size_bias,                  cudaMemcpyHostToDevice); 
    cudaMemset(contribution_matrix_d,   0,              size_contribution_matrix);

    int offset_alive_cell = 0;
    while(offset_alive_cell<alive_cell){

        int max = numero_workspace<alive_cell-offset_alive_cell?numero_workspace:alive_cell-offset_alive_cell;

        for(int workspace_idx=0; workspace_idx<max; workspace_idx++){

            int offset_workspace_in = input_dim*workspace_idx;
            int offset_workspace_out = output_dim*workspace_idx;

            launch_vision(
                world_value_d,
                world_id_d,
                world_signal_d,
                world_dim,
                alive_cell_d+offset_alive_cell,
                visione,
                inputs_d+offset_workspace_in,
                stream
            );

            launch_NN_forward(
                inputs_d+offset_workspace_in,
                inputs_d+offset_workspace_in,
                weight_d,
                n_weights,
                baias_d,
                n_baias,
                structure,
                offset_alive_cell,
                alive_cell_d,
                world_id_d,
                n,
                stream
            );

            launch_output_elaboration(
                world_value_d,
                world_signal_d,
                world_id_d,
                contribution_matrix_d,
                inputs_d+offset_workspace_in,
                alive_cell_d,
                world_dim,
                n_modelli,
                output_dim,
                offset_alive_cell,
                stream
            );

            offset_alive_cell++;

        }

    }

    cudaMemcpy(world_value_h,           world_value_d,          size_world,                 cudaMemcpyDeviceToHost);
    cudaMemcpy(world_signal_h,          world_signal_d,         size_world,                 cudaMemcpyDeviceToHost);
    cudaMemcpy(world_id_h,              world_id_d,             size_world,                 cudaMemcpyDeviceToHost);
    cudaMemcpy(alive_cell_h,            alive_cell_d,           size_world,                 cudaMemcpyDeviceToHost);
    cudaMemcpy(inputs_h,                inputs_d,               size_inputs,                cudaMemcpyDeviceToHost); 
    cudaMemcpy(outputs_h,               outputs_d,              size_outputs,               cudaMemcpyDeviceToHost);
    cudaMemcpy(weight_h,                weight_d,               size_weight,                cudaMemcpyDeviceToHost);
    cudaMemcpy(baias_h,                 baias_d,                size_bias,                  cudaMemcpyDeviceToHost); 
    cudaMemcpy(contribution_matrix_h,   contribution_matrix_d,  size_contribution_matrix,   cudaMemcpyDeviceToHost); 


    std::cout << "\n=== WORLD VALUE ===\n";
    for (int y = 0; y < world_dim; y++) {
        for (int x = 0; x < world_dim; x++) {
            printf("%.4f ", world_value_h[y * world_dim + x]);
        }
        std::cout << "\n";
    }

    std::cout << "\n=== WORLD ID ===\n";
    for (int y = 0; y < world_dim; y++) {
        for (int x = 0; x < world_dim; x++) {
            printf("%4d ",world_id_h[y * world_dim + x]);
        }
        std::cout << "\n";
    }

    std::cout << "\n=== WORLD SIGNAL ===\n";
    for (int y = 0; y < world_dim; y++) {
        for (int x = 0; x < world_dim; x++) {
            printf("%.4f ",world_signal_h[y * world_dim + x]);
        }
        std::cout << "\n";
    }

    std::cout << "\n=== ALIVE CELLS ===\n";
    for (int i = 0; i < alive_cell; i++) {
        std::cout << "Alive[" << i << "] = " << alive_cell_h[i] << "\n";
    }

    std::cout << "\n=== INPUTS (workspace x input_dim) ===\n";
    for (int ws = 0; ws < numero_workspace; ws++) {
        std::cout << "Workspace " << ws << ": \n";
        for (int j = 0; j < input_dim; j++) {
            printf("%.4f ",inputs_h[ws * input_dim + j]);
        }
        std::cout << "\n";
    }


    std::cout << "\n=== OUTPUTS (workspace x input_dim) ===\n";
    for (int ws = 0; ws < numero_workspace; ws++) {
        std::cout << "Workspace " << ws << ": \n";
        for (int j = 0; j < output_dim; j++) {
            printf("%.4f ",outputs_h[ws * output_dim + j]);
        }
        std::cout << "\n";
    }

    std::cout << "\n=== CONTRIBUTION MATRIX===\n";
    for(int i=0; i<world_dim; i++){
        for(int j=0; j<world_dim; j++){
            printf("( ");
            for(int k=0; k<n_modelli; k++){
                printf("%.4f ",contribution_matrix_h[(i * world_dim) + j + (k * world_dim*world_dim) ]);
            }
            printf(") ");
        }
        printf("\n");
    }
    printf("\n");

    printf("=== MODELLO ===\n");
    printf("BIASES :  %d\n", n_baias);
    printf("WEIGHTS:  %d\n", n_weights);

    printf("=== DEVICE POINTERS ===\n");
    printf("WORLD VALUE  (float*): %p\n", (void*)world_value_d);
    printf("WORLD SIGNAL (float*): %p\n", (void*)world_signal_d);
    printf("WORLD ID     (int*)  : %p\n", (void*)world_id_d);
    printf("ALIVE CELL   (int*)  : %p\n", (void*)alive_cell_d);
    printf("INPUTS       (float*): %p\n", (void*)inputs_d);    
    printf("OUTPUTS      (float*): %p\n", (void*)outputs_d);   
    printf("WEIGHTS      (float*): %p\n", (void*)weight_d);   
    printf("BIASES       (float*): %p\n", (void*)baias_d); 
    printf("CONTRIBUTION (float*): %p\n", (void*)contribution_matrix_d);   


    return 0;

}