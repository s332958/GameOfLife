#include <cuda_runtime.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <stdexcept>

inline void checkCudaError(cudaError_t result, const char *msg, const char *file, int line) {
    if (result != cudaSuccess) {
        std::string error_message = std::string("[CUDA ERROR] ") + msg + 
            " at " + file + ":" + std::to_string(line) + 
            "\n  → " + cudaGetErrorString(result);
        throw std::runtime_error(error_message);
        std::cout<<error_message<<"\n";
    }
}

#define CUDA_CHECK(val) checkCudaError((val), #val, __FILE__, __LINE__)


// dunction for obtain the free GPU memory value
void computeFreeMemory(size_t *free_memory) {
    size_t t;
    cudaMemGetInfo(free_memory, &t);
}

// Function for allocate syncronous or asyncronous memory, in base at cc
void* cuda_allocate(size_t size, int cc_major, cudaStream_t stream = 0) {
    void* ptr = nullptr;
    cudaError_t err;

    if (cc_major >= 7) {
        err = cudaMallocAsync(&ptr, size, stream);
    } else {
        err = cudaMalloc(&ptr, size);
    }

    if (err != cudaSuccess) {
        std::cerr << "CUDA allocation failed: " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }

    return ptr;
}

// Function for copy syncronous or asyncronous memory, in base at cc
void cuda_memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind, int cc_major, cudaStream_t stream = 0) {
    cudaError_t err;

    if (cc_major >= 7) {
        // Copia asincrona su stream specificato
        err = cudaMemcpyAsync(dst, src, size, kind, stream);
    } else {
        // Copia sincrona (ignora lo stream)
        err = cudaMemcpy(dst, src, size, kind);
    }

    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}


// Function for deallocate syncronous or asyncronous memory, in base at cc
void cuda_Free(void* ptr, int cc_major, cudaStream_t stream) {
    if (cc_major >= 7) {
        cudaFreeAsync(ptr, stream);
    } else {
        cudaFree(ptr);
    }
}

// Function for compute next stream value
int next_stream(int *index_stream, int limit){
    *index_stream = ((*index_stream)+1) % limit;
    return *index_stream;
}

// Function for generate a random value between a min and max
int get_random_int(int min, int max) {
    static std::random_device rd;                      
    static std::mt19937 gen(rd());                    
    std::uniform_int_distribution<> dis(min, max); 

    return dis(gen);
}

// Function to save models on file
void save_model_on_file(
    const std::string& nome_file,
    const float* weights,
    const float* biases,
    int dim_pesi,
    int dim_bias
)
{
    std::ofstream out(nome_file);
    if (!out.is_open()) {
        throw std::runtime_error("Impossible open the file for writing.");
    }


    for (int i = 0; i < dim_pesi; ++i) {
        out << weights[i];
        if (i < dim_pesi - 1) out << " ";
    }
    out << "\n";

    for (int i = 0; i < dim_bias; ++i) {
        out << biases[i];
        if (i < dim_bias - 1) out << " ";
    }

    out.close();
}

// function for loading models from file
int load_model_from_file(
    const std::string& nome_file,
    float* weights_host,
    float* biases_host,
    int dim_pesi,
    int dim_bias
) {
    std::ifstream in(nome_file);
    if (!in.is_open()) {
        std::cout << "Impossible open the file for reading: " << nome_file << std::endl;
        return 0;
    }

    std::string line;

    // Riga 1: pesi
    std::getline(in, line);
    std::istringstream pesi_stream(line);
    for (int i = 0; i < dim_pesi; ++i) {
        if (!(pesi_stream >> weights_host[i])) {
            std::cerr << "Error in reading weights (indice " << i << ")" << std::endl;
            return 0;
        }
    }

    // Riga 2: bias
    std::getline(in, line);
    std::istringstream bias_stream(line);
    for (int i = 0; i < dim_bias; ++i) {
        if (!(bias_stream >> biases_host[i])) {
            std::cerr << "Error in reading biases (indice " << i << ")" << std::endl;
            return 0;
        }
    }

    in.close();
    return 1;
}

// file for saving global score 
void append_score_to_file(const std::string& filename, float tot_score) {
    std::ofstream out(filename, std::ios::app);  
    if (!out.is_open()) {
        throw std::runtime_error("Impossible to open the file.");
    }

    out << tot_score << "\n";  
    out.close();
}
