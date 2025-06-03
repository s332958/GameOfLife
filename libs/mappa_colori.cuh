// mappa_colori.cuh
#pragma once

#include <cuda_runtime.h>

// Function for init GPU constant memory for the colors
void load_constant_memory_GPU(int n_creature);

// Function for launch world color mapping
void launch_mappa_colori(float* mondo, int* id_matrix, float* mondo_rgb_d, int world_dim, cudaStream_t stream);

// Function for launch world signal color mapping
void launch_mappa_signal(float* mondo, int* id_matrix, float* mondo_signal, float* mondo_rgb_d, int world_dim, int n_creature, cudaStream_t stream);

// GPU Zone for colors (in constant memory)
extern __constant__ float COLORI[100][3];
