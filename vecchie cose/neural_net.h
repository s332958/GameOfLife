#pragma once
#include <cuda_runtime.h>

struct NeuralNet {
    int numLayers;
    int* h_layerSizes;  // host

    float* h_weights;
    float* h_biases;

    int totalWeights;
    int totalBiases;

    int totalParams;

    NeuralNet(int* sizes, int nLayers, float* allParams, int totW, int totB);
    ~NeuralNet();    
};

