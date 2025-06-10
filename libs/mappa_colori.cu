
 // #include "mappa_colori.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>

// Declare constant memory for save color matrix using for map the color during the rendering
__constant__ float COLORI[300][3];


// declere color in host 
void generateDistinctColors(float* colors, int n_creature) {
    for (int i = 0; i < n_creature; ++i) {
        // HSV -> RGB conversion per generare colori distinti
        float angle = i * 360.0f / n_creature + 180.0f;

        // subtract off however many full turns (360°) it contains
        float h = angle - floorf(angle / 360.0f) * 360.0f;
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


// Save color in GPU after generating them in GPU
void load_constant_memory_GPU(int n_creature) {
    float *color_h = (float*) malloc(sizeof(int)*n_creature*3);
    generateDistinctColors(color_h, n_creature);

    // to constant memory
    cudaMemcpyToSymbol(COLORI, color_h, sizeof(float) * 300 * 3);
    free(color_h);
}


// Kernel for generating colors from world_value and world_id
__global__ void mappa_colori_kernel(float* mondo, int* id_matrix, float* mondo_rgb, int thread_number) {
    
    // number of thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // return if exceed
    if (index >= thread_number) return;

    //compute pixel index and channel index 
    int pixel_index = index / 3;
    int channel = index % 3;

    // get id from id_matrix and get value from world_value (mondo)
    int ID = id_matrix[pixel_index];
    float value = mondo[pixel_index];

    // compute index output
    int out_index = pixel_index * 3 + channel;

    // id obstacle chise color red
    if (ID == -1) {
        mondo_rgb[out_index] = (channel == 0) ? 1.0f : 0.0f;  
    }
    // neutral cell are in gray scale
    else if (ID == 0) {
        mondo_rgb[out_index] = value;  
    }
    // for other color read the constant memory with colors
    else{
        mondo_rgb[out_index] = COLORI[ID - 1][channel] * (value + 0.005) * 5;
    }

}

// similar to the precedent but get also the signal map
__global__ void mappa_signal_kernel(float * mondo_value, int* id_matrix, float* mondo_signal, float* mondo_rgb, int n_creature, int thread_number) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= thread_number) return;

    int pixel_index = index / 3;
    int channel = index % 3;

    int ID = id_matrix[pixel_index];
    float value = mondo_signal[pixel_index];
    float signal = value;

    int out_index = pixel_index * 3 + channel;

    if (ID == -1) {
        mondo_rgb[out_index] = (channel == 0) ? 1.0f : 0.0f;  
    }
    else if (ID == 0) {
        mondo_rgb[out_index] = mondo_value[pixel_index]; 
    }
    else{
        mondo_rgb[out_index] = COLORI[ID - 1][channel] * signal;
    }

}


void launch_mappa_colori(float* mondo, int* id_matrix, float* mondo_rgb_d, int world_dim, cudaStream_t stream){

    int n_thread_per_block = 1024; 
    int thread_number = world_dim*world_dim*3;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;


    mappa_colori_kernel<<<n_block, n_thread_per_block, 0, stream>>>(mondo, id_matrix, mondo_rgb_d, thread_number);

}

void launch_mappa_signal(float* mondo, int* id_matrix, float* mondo_signal, float* mondo_rgb_d, int world_dim, int n_creature, cudaStream_t stream){

    int n_thread_per_block = 1024; 
    int thread_number = world_dim*world_dim*3;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;


    mappa_signal_kernel<<<n_block, n_thread_per_block, 0, stream>>>(mondo, id_matrix, mondo_signal, mondo_rgb_d, n_creature, thread_number);

}