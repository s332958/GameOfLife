// mappa_colori.cuh
#pragma once

#include <cuda_runtime.h>

// Funzione per inizializzare la memoria costante sulla GPU
void load_constant_memory_GPU();

// Funzione che lancia il kernel di color mapping
void launch_mappa_colori(float* mondo, int* id_matrix, float* mondo_rgb_d, int world_dim, cudaStream_t stream);
