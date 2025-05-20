#ifndef UTILS
#define UTILS

#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <random>
#include <fstream>
#include <string>
/*
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
*/

// Funzione per ottenere memoria libera disponibile sulla GPU
void computeFreeMemory(size_t *free_memory) {
    size_t t;
    cudaMemGetInfo(free_memory, &t);
}

// Funzione per allocare memoria sulla GPU, sincrona o asincrona in base alla compute capability
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

// Funzione per spostare i dati 
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


// Funzione per deallocare memoria sulla GPU, sincrona o asincrona in base alla compute capability
void cuda_Free(void* ptr, int cc_major, cudaStream_t stream) {
    if (cc_major >= 7) {
        cudaFreeAsync(ptr, stream);
    } else {
        cudaFree(ptr);
    }
}

// Funzione per mantenere l'ordine degli stream
int next_stream(int *index_stream, int limit){
    *index_stream = ((*index_stream)+1) % limit;
    return *index_stream;
}

// Funzione per generare numero random tra min e max
int get_random_int(int min, int max) {
    static std::random_device rd;                      // sorgente di entropia
    static std::mt19937 gen(rd());                     // generatore Mersenne Twister
    std::uniform_int_distribution<> dis(min, max); // intervallo [0, limit-1]

    return dis(gen);
}

// Funzione per salvare i modelli su file
void save_model_on_file(
    const std::string& nome_file,
    const int* dim,
    int dim_size,
    const float* pesi_totale,
    const float* bias_totale,
    int dim_pesi,
    int dim_bias,
    int n_modelli)
{
    std::ofstream out(nome_file, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("Impossibile aprire il file per la scrittura.");
    }

    for (int modello = 0; modello < n_modelli; ++modello) {
        // Riga 1: vettore dim
        for (int i = 0; i < dim_size; ++i) {
            out << dim[i];
            if (i < dim_size - 1) out << " ";
        }
        out << "\n";

        // Riga 2: pesi
        const float* pesi_inizio = pesi_totale + modello * dim_pesi;
        for (int i = 0; i < dim_pesi; ++i) {
            out << pesi_inizio[i];
            if (i < dim_pesi - 1) out << " ";
        }
        out << "\n";

        // Riga 3: bias
        const float* bias_inizio = bias_totale + modello * dim_bias;
        for (int i = 0; i < dim_bias; ++i) {
            out << bias_inizio[i];
            if (i < dim_bias - 1) out << " ";
        }
        out << "\n";

        out << "\n";out << "\n";out << "\n";
    }

    out.close();
}


// Funzione per salvare mappa su file
void save_map(
    FILE *file,
    const int dim_world,
    const float* world_value,
    const int* world_id
){

    if (file == NULL) {
        perror("Errore nell'apertura del file");
        return;
    }


    for(int riga=0; riga<dim_world; riga++){
        for(int colonna=0; colonna<dim_world; colonna++){
            fprintf(file,"%.2f %3d ",world_value[riga*dim_world+colonna],world_id[riga*dim_world+colonna]);
        }
        fprintf(file,"\n");
    }
    // fprintf(file,"\n\n\n");

}









inline void checkCudaError(cudaError_t result, const char *msg, const char *file, int line) {
    if (result != cudaSuccess) {
        std::string error_message = std::string("[CUDA ERROR] ") + msg + 
            " at " + file + ":" + std::to_string(line) + 
            "\n  → " + cudaGetErrorString(result);
        //throw std::runtime_error(error_message);
        std::cout<<error_message<<"\n";
    }
}

#define CUDA_CHECK(val) checkCudaError((val), #val, __FILE__, __LINE__)


// FUNZIONI THRUST

// Funzione template per sorting decrescente degli indici
/*
template <typename T>
void sort_indices_by_values_desc(T* d_values, int* d_sorted_indices, int N) {
    // Rimuove const solo per poter usare sort_by_key (non modifica i dati)
    thrust::device_ptr<T> values_ptr(const_cast<T*>(d_values));
    thrust::device_ptr<int> indices_ptr(d_sorted_indices);

    thrust::sequence(indices_ptr, indices_ptr + N);
    thrust::sort_by_key(values_ptr, values_ptr + N, indices_ptr, thrust::greater<T>());
}
*/


// Predicate: true se il valore è > 0
struct is_positive {
    __host__ __device__
    bool operator()(int x) const {
        return x > 0;
    }
};

// Compattta d_input[0..N-1], restituisce:
// - d_output (da allocare esternamente, capienza almeno N)
// - num_validi: numero effettivo di elementi copiati
// si puo usare input come output poiche thrust lavora in modo sequenziale
/*
void compact_positive(int* d_input, int* d_output, int N, int* dim_output_h) {
    // Wrappa i puntatori raw con thrust::device_ptr
    auto input_begin = thrust::device_pointer_cast(d_input);
    auto input_end   = input_begin + N;
    auto output_begin = thrust::device_pointer_cast(d_output);

    // Esegui compattamento
    auto output_end = thrust::copy_if(
        thrust::device,
        input_begin, input_end,
        output_begin,
        is_positive()
    );

    // Calcola numero di elementi copiati
    *dim_output_h = output_end - output_begin;
}
*/

void argsort_bubble(float *vettore, int *indice, int n) {
    for (int i = 0; i < n; i++) indice[i] = i;

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (vettore[indice[j]] < vettore[indice[j + 1]]) {
                int temp = indice[j];
                indice[j] = indice[j + 1];
                indice[j + 1] = temp;
            }
        }
    }
}

#endif