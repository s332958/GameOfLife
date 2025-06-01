#pragma once

#include <cstddef>
#include <string>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

// ========== checkCudaError ==========
inline void checkCudaError(cudaError_t result, const char* msg, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " -> " << msg << ": " << cudaGetErrorString(result) << "\n";
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(val) checkCudaError((val), #val, __FILE__, __LINE__)

// ========== TEMPLATE argsort_bubble ==========
template <typename T>
void argsort_bubble(T* vettore, int* indice, int n) {
    for (int i = 0; i < n; i++) indice[i] = i;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (vettore[indice[j]] < vettore[indice[j + 1]]) {
                int tmp = indice[j];
                indice[j] = indice[j + 1];
                indice[j + 1] = tmp;
            }
        }
    }
}

// ========== DICHIARAZIONI altre funzioni ==========
void computeFreeMemory(size_t* free_memory);
void* cuda_allocate(size_t size, int cc_major, cudaStream_t stream = 0);
void cuda_memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind, int cc_major, cudaStream_t stream = 0);
void cuda_Free(void* ptr, int cc_major, cudaStream_t stream);
int next_stream(int* index_stream, int limit);
int get_random_int(int min, int max);
void save_model_on_file(const std::string& nome_file, const int* dim, int dim_size,
                        const float* pesi_totale, const float* bias_totale, int dim_pesi, int dim_bias, int n_modelli);
void load_model_from_file(const std::string& nome_file, float* pesi_totale, float* bias_totale, int dim_pesi, int dim_bias, int n_modelli);
void save_map(FILE* file, int dim_world, const float* world_value, const int* world_id);
