#include <iostream>
#include <cstdlib>
#include <ctime>
#include "libs/neural_net.cuh"
#include <cuda_runtime.h>
#include <random>

int main() {
    // Inizializzo dimensioni rete dinamicamente
    int numLayers = 3;
    int* sizes = new int[numLayers];
    sizes[0] = 4;
    sizes[1] = 3;
    sizes[2] = 2;

    int totalWeights = sizes[0]*sizes[1] + sizes[1]*sizes[2]; // 12 + 6 = 18
    int totalBiases = sizes[1] + sizes[2]; // 3 + 2 = 5
    int totalParams = totalWeights + totalBiases;

    // Alloco e inizializzo parametri rete
    float* allParams = new float[totalParams];
    srand((unsigned)time(nullptr));
    for (int i = 0; i < totalParams; ++i)
        allParams[i] = static_cast<float>(rand()) / RAND_MAX;

    // Creo NeuralNet con parametri dinamici
    NeuralNet net(sizes, numLayers, allParams);

    // Input di test
    float input[4] = {1.0f, 0.5f, -0.5f, 0.2f};
    float output[2] = {0};

    net.forwardOnDevice(input, output);

    std::cout << "Output rete dopo forwardOnDevice:\n";
    for (int i = 0; i < 2; i++) std::cout << output[i] << " ";
    std::cout << std::endl;

    // Test argsort_bubble su energy (float) e coverage (int)
    int nNets = 5;
    float energies[5] = {0.9f, 0.3f, 0.5f, 0.7f, 0.1f};
    int coverages[5] = {10, 50, 20, 30, 40};

    int indicesEnergy[5];
    int indicesCoverage[5];

    argsort_bubble(energies, indicesEnergy, nNets);
    argsort_bubble(coverages, indicesCoverage, nNets);

    std::cout << "Indices sorted by energy (desc): ";
    for (int i = 0; i < nNets; i++) std::cout << indicesEnergy[i] << " ";
    std::cout << std::endl;

    std::cout << "Indices sorted by coverage (desc): ";
    for (int i = 0; i < nNets; i++) std::cout << indicesCoverage[i] << " ";
    std::cout << std::endl;

    // Creo array NeuralNet per test recombination
    NeuralNet* neuralNets = (NeuralNet* ) malloc(nNets*sizeof(NeuralNet));
    NeuralNet* newNeuralNets = (NeuralNet* ) malloc(nNets*sizeof(NeuralNet));

    // Inizializzo neuralNets copiando parametri uguali (solo per test)
    for (int i = 0; i < nNets; i++) {
        neuralNets[i] = NeuralNet(sizes, numLayers, allParams);
        newNeuralNets[i] = NeuralNet(sizes, numLayers, allParams);
    }

    // CUDA stream (default 0)
    cudaStream_t stream = 0;

    // Chiamata a wrapper_recombination
    float limit = 0.6f; // usa il 60% delle reti
    int type_of_union = 0; // dummy, non usato ora
    int random_mutation_x_block = 2;
    int dim_block = 32;
    float max_variation_mutation = 0.1f;

    try {
        wrapper_recombination(neuralNets, newNeuralNets, nNets, energies, coverages, limit,
                             type_of_union, random_mutation_x_block, dim_block, max_variation_mutation, stream);
    } catch (const char* e) {
        std::cerr << "Errore: " << e << std::endl;
    }

    std::cout << "Recombination kernel lanciato." << std::endl;

    // Pulizia
    net.clear();
    for (int i = 0; i < nNets; i++) {
        neuralNets[i].clear();
        newNeuralNets[i].clear();
    }
    delete[] neuralNets;
    delete[] newNeuralNets;
    delete[] sizes;
    delete[] allParams;

    return 0;
}
