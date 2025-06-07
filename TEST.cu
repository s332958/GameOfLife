#include <cstdio>
#include <curand_kernel.h>
#include "libs/utils_kernel.cu"  // Assicurati che i kernel siano definiti qui

int main() {
    // Parametri
    int n_biases = 2;
    int n_weights = 10;
    int n_creation = 10;
    float std = 1.0f;
    float alpha = 0.01f;
    int n_threads = 1024;

    // Puntatori device (GPU)
    float *random_weights_d = nullptr;
    float *random_biases_d = nullptr;
    float *new_weights_d = nullptr;
    float *new_biases_d = nullptr;
    float *variation_weights_d = nullptr;
    float *variation_biases_d = nullptr;
    float *score_value_d = nullptr;
    curandState *states_d = nullptr;

    // Allocazioni device
    cudaMalloc((void**)&random_weights_d, n_weights * sizeof(float));
    cudaMalloc((void**)&random_biases_d, n_biases * sizeof(float));
    cudaMalloc((void**)&new_weights_d, n_weights * n_creation * sizeof(float));
    cudaMalloc((void**)&new_biases_d, n_biases * n_creation * sizeof(float));
    cudaMalloc((void**)&variation_weights_d, n_weights * n_creation * sizeof(float));
    cudaMalloc((void**)&variation_biases_d, n_biases * n_creation * sizeof(float));
    cudaMalloc((void**)&score_value_d, n_creation * sizeof(float));
    cudaMalloc((void**)&states_d, n_threads * sizeof(curandState));

    // Inizializzazione curand
    launch_init_curandstates(states_d, n_threads, 0, 0);

    // Riempimento iniziale random
    launch_fill_random_kernel(random_weights_d, 0, n_weights, -1, 1, 0, 0);
    launch_fill_random_kernel(random_biases_d, 0, n_biases, -1, 1, 0, 0);
    launch_fill_random_kernel(score_value_d,0,n_creation,-1,1,0,0);

    // Generazione creature
    launch_generate_clone_creature(
        random_weights_d, random_biases_d,
        new_weights_d, new_biases_d,
        variation_weights_d, variation_biases_d,
        n_weights, n_biases, n_creation,
        std, 0, states_d
    );

    // Update modello
    launch_update_model(
        random_weights_d, random_biases_d,
        variation_weights_d, variation_biases_d,
        score_value_d,
        n_weights, n_biases, n_creation,
        alpha, std, 0
    );

    // --- ALLOCAZIONE HOST (CPU) ---
    float *h_random_weights = new float[n_weights];
    float *h_random_biases = new float[n_biases];
    float *h_new_weights = new float[n_weights * n_creation];
    float *h_new_biases = new float[n_biases * n_creation];
    float *h_variation_weights = new float[n_weights * n_creation];
    float *h_variation_biases = new float[n_biases * n_creation];
    float *h_score_value = new float[n_creation];

    // --- COPIA DA DEVICE A HOST ---
    cudaMemcpy(h_random_weights, random_weights_d, n_weights * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_random_biases, random_biases_d, n_biases * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_new_weights, new_weights_d, n_weights * n_creation * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_new_biases, new_biases_d, n_biases * n_creation * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variation_weights, variation_weights_d, n_weights * n_creation * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variation_biases, variation_biases_d, n_biases * n_creation * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_score_value, score_value_d, n_creation * sizeof(float), cudaMemcpyDeviceToHost);

    // --- STAMPA DATI ---
    printf("Random Weights:\n");
    for (int i = 0; i < n_weights; ++i)
        printf("%.4f ", h_random_weights[i]);
    printf("\n\n");

    printf("Random Biases:\n");
    for (int i = 0; i < n_biases; ++i)
        printf("%.4f ", h_random_biases[i]);
    printf("\n\n");

    printf("New Weights (1st individual):\n");
    for (int i = 0; i < n_weights; ++i)
        printf("%.4f ", h_new_weights[i]);
    printf("\n\n");

    printf("Variation Weights (1st individual):\n");
    for (int i = 0; i < n_weights; ++i)
        printf("%.4f ", h_variation_weights[i]);
    printf("\n\n");

    printf("Score Values:\n");
    for (int i = 0; i < n_creation; ++i)
        printf("%.4f ", h_score_value[i]);
    printf("\n");

    // --- DEALLOCAZIONE ---
    cudaFree(random_weights_d);
    cudaFree(random_biases_d);
    cudaFree(new_weights_d);
    cudaFree(new_biases_d);
    cudaFree(variation_weights_d);
    cudaFree(variation_biases_d);
    cudaFree(score_value_d);
    cudaFree(states_d);

    delete[] h_random_weights;
    delete[] h_random_biases;
    delete[] h_new_weights;
    delete[] h_new_biases;
    delete[] h_variation_weights;
    delete[] h_variation_biases;
    delete[] h_score_value;

    return 0;
}
