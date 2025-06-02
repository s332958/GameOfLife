
 // #include "mappa_colori.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>

// Dichiaro la memoria costante in GPU dove verranno tenuti i colori per poi fare il rendering
__constant__ float COLORI[100][3];


// Dichiaro i colori in host
void generateDistinctColors(float* colors, int n_creature) {
    for (int i = 0; i < n_creature; ++i) {
        // HSV -> RGB conversion per generare colori distinti
        float h = (i * 360.0f / n_creature);
        float s = 1.0f;
        float v = 1.0f;

        float c = v * s;
        float x = c * (1 - fabs(fmod(h / 60.0f, 2) - 1));
        float m = v - c;

        float r, g, b;
        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        colors[i*3+0] = r+m;
        colors[i*3+1] = g+m;
        colors[i*3+2] = b+m;

    }
}


// Passo i colori in GPU e poi libero l'allocazione in Host
void load_constant_memory_GPU(int n_creature) {
    float *color_h = (float*) malloc(sizeof(int)*n_creature*3);
    generateDistinctColors(color_h, n_creature);


    cudaMemcpyToSymbol(COLORI, color_h, sizeof(float) * 100 * 3);
    free(color_h);
}

// Kernel per generare i colori corrispettivi dati mondo_value, monod_id, e l'output mondo_rgb
__global__ void mappa_colori_kernel(float* mondo, int* id_matrix, float* mondo_rgb, int thread_number) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= thread_number) return;

    int pixel_index = index / 3;
    int channel = index % 3;

    int ID = id_matrix[pixel_index];
    float value = mondo[pixel_index];

    int out_index = pixel_index * 3 + channel;

    if (ID == -1) {
        mondo_rgb[out_index] = (channel == 0) ? 1.0f : 0.0f;  // rosso per ostacoli
    }
    else if (ID == 0) {
        mondo_rgb[out_index] = value;  // scala di grigi
    }
    else{
        mondo_rgb[out_index] = COLORI[ID - 1][channel] * (value + 0.01) * 5;
    }

}

__global__ void mappa_signal_kernel(float * mondo_value, int* id_matrix, float* mondo_signal, float* mondo_rgb, int n_creature, int thread_number) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= thread_number) return;

    int pixel_index = index / 3;
    int channel = index % 3;

    int ID = id_matrix[pixel_index];
    float value = mondo_signal[pixel_index];
    float signal = value;
    //(value - (1/n_creature)*(ID-1))*n_creature;


    int out_index = pixel_index * 3 + channel;

    if (ID == -1) {
        mondo_rgb[out_index] = (channel == 0) ? 1.0f : 0.0f;  // rosso per ostacoli
    }
    else if (ID == 0) {
        mondo_rgb[out_index] = mondo_value[pixel_index];  // scala di grigi
    }
    else{
        mondo_rgb[out_index] = COLORI[ID - 1][channel] * (signal + 0.02) * 3;
    }

}


void launch_mappa_colori(float* mondo, int* id_matrix, float* mondo_rgb_d, int world_dim, cudaStream_t stream){

    int n_thread_per_block = 1024; //properties.maxThreadsPerBlock; 
    int thread_number = world_dim*world_dim*3;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;


    mappa_colori_kernel<<<n_block, n_thread_per_block, 0, stream>>>(mondo, id_matrix, mondo_rgb_d, thread_number);

}

void launch_mappa_signal(float* mondo, int* id_matrix, float* mondo_signal, float* mondo_rgb_d, int world_dim, int n_creature, cudaStream_t stream){

    int n_thread_per_block = 1024; //properties.maxThreadsPerBlock; 
    int thread_number = world_dim*world_dim*3;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;


    mappa_signal_kernel<<<n_block, n_thread_per_block, 0, stream>>>(mondo, id_matrix, mondo_signal, mondo_rgb_d, n_creature, thread_number);

}