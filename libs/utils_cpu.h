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

// ======== OTHER FUNCTION =======================
void computeFreeMemory(size_t* free_memory);
void* cuda_allocate(size_t size, int cc_major, cudaStream_t stream = 0);
void cuda_memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind, int cc_major, cudaStream_t stream = 0);
void cuda_Free(void* ptr, int cc_major, cudaStream_t stream);
int  next_stream(int* index_stream, int limit);
int  get_random_int(int min, int max);
void save_model_on_file(const std::string& nome_file, const float* pesi_totale, const float* bias_totale, int dim_pesi, int dim_bias);
int  load_model_from_file(const std::string& nome_file, float* pesi_totale, float* bias_totale, int dim_pesi, int dim_bias);
void append_score_to_file(const std::string& filename, float tot_score);
