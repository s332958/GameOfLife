#include "neural_net.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cstring>

NeuralNet::NeuralNet(int* sizes, int nLayers, float* allParams)
    : numLayers(nLayers) {
    assert(nLayers >= 2);

    h_layerSizes = new int[numLayers];
    std::memcpy(h_layerSizes, sizes, sizeof(int) * numLayers);

    cudaMalloc(&d_layerSizes, sizeof(int) * numLayers);
    cudaMemcpy(d_layerSizes, sizes, sizeof(int) * numLayers, cudaMemcpyHostToDevice);

    totalWeights = 0;
    totalBiases = 0;
    for (int i = 1; i < numLayers; ++i) {
        totalWeights += sizes[i - 1] * sizes[i];
        totalBiases += sizes[i];
    }

    cudaMalloc(&d_weights, sizeof(float) * totalWeights);
    cudaMalloc(&d_biases, sizeof(float) * totalBiases);

    cudaMemcpy(d_weights, allParams, sizeof(float) * totalWeights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, allParams + totalWeights, sizeof(float) * totalBiases, cudaMemcpyHostToDevice);
}

NeuralNet::~NeuralNet() {
    clear();
}

void NeuralNet::clear() {
    delete[] h_layerSizes;
    cudaFree(d_layerSizes);
    cudaFree(d_weights);
    cudaFree(d_biases);

    h_layerSizes = nullptr;
    d_layerSizes = nullptr;
    d_weights = nullptr;
    d_biases = nullptr;
    numLayers = 0;
    totalWeights = 0;
    totalBiases = 0;
}

void NeuralNet::forwardOnDevice(const float* h_input, float* h_output) {
    const int inputSize = h_layerSizes[0];
    const int outputSize = h_layerSizes[numLayers - 1];

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, sizeof(float) * inputSize);
    cudaMalloc(&d_output, sizeof(float) * outputSize);

    cudaMemcpy(d_input, h_input, sizeof(float) * inputSize, cudaMemcpyHostToDevice);

    int threads = 256;
    int maxLayerSize = 0;
    for (int i = 0; i < numLayers; ++i)
        if (h_layerSizes[i] > maxLayerSize)
            maxLayerSize = h_layerSizes[i];

    size_t sharedMem = sizeof(float) * 2 * maxLayerSize;
    forward_kernel<<<1, maxLayerSize, sharedMem>>>(
        d_input, d_output,
        d_layerSizes, d_weights, d_biases, numLayers
    );

    cudaMemcpy(h_output, d_output, sizeof(float) * outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}





__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__global__ void forward_kernel(const float* input, float* output,
                               const int* layerSizes,
                               const float* weights, const float* biases,
                               int numLayers) {
    extern __shared__ float shared[];
    float* current = shared;
    float* next = shared + layerSizes[0];

    int tid = threadIdx.x;

    if (tid < layerSizes[0])
        current[tid] = input[tid];
    __syncthreads();

    int wOffset = 0;
    int bOffset = 0;

    for (int l = 1; l < numLayers; ++l) {
        int inSize = layerSizes[l - 1];
        int outSize = layerSizes[l];

        if (tid < outSize) {
            float sum = biases[bOffset + tid];
            for (int j = 0; j < inSize; ++j)
                sum += current[j] * weights[wOffset + tid * inSize + j];
            next[tid] = relu(sum);
        }

        __syncthreads();
        float* temp = current;
        current = next;
        next = temp;

        wOffset += inSize * outSize;
        bOffset += outSize;
    }

    if (tid < layerSizes[numLayers - 1])
        output[tid] = current[tid];
}
