#pragma once

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
