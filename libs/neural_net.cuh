#pragma once
#include <cuda_runtime.h>

struct NeuralNet {
    int numLayers;
    int* h_layerSizes;  // host
    int* d_layerSizes;  // device

    float* d_weights;
    float* d_biases;

    int totalWeights;
    int totalBiases;

    NeuralNet(int* sizes, int nLayers, float* allParams);
    ~NeuralNet();

    void clear();
    void forwardOnDevice(const float* h_input, float* h_output);
    
};


void argsort_bubble(int *vettore, int *indice, int n);
void argsort_bubble(float *vettore, int *indice, int n);

void wrapper_recombination(NeuralNet *neuralNets, NeuralNet *newNeuralNet, int totNeuralNet, float *total_energy, int *total_coverage, float limit, 
    int type_of_union, int random_mutation_x_block, int dim_block, float max_varaiation_mutation,cudaStream_t stream);

__global__ void recombination(NeuralNet n1, NeuralNet n2, NeuralNet final, int n_random_mutation_x_block, float max_value_mutation, int totalWeights, int totalBiases);
