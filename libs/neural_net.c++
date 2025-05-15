#include "neural_net.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cstring>

NeuralNet::NeuralNet(int* sizes, int nLayers, float* allParams, int totalWeights, int totalBiases)
: numLayers(nLayers) {

    assert(sizes != nullptr);
    assert(allParams != nullptr);
    assert(nLayers >= 2);


    h_layerSizes = new int[numLayers];
    std::memcpy(h_layerSizes, sizes, sizeof(int) * numLayers);


    totalParams = totalWeights + totalBiases;

    h_weights = new float[totalWeights];
    h_biases = new float[totalBiases];
    std::memcpy(h_weights, allParams, sizeof(float) * totalWeights);
    std::memcpy(h_biases, allParams + totalWeights, sizeof(float) * totalBiases);
}

NeuralNet::~NeuralNet() {
    delete[] h_layerSizes;
    delete[] h_weights;
    delete[] h_biases;
}

