#include <cuda_runtime.h>

#ifndef KERNEL
#define KERNEL


void argsort_bubble(int *vettore, int *indice, int n);
void argsort_bubble(float *vettore, int *indice, int n);

void wrapper_recombination(NeuralNet *neuralNets, NeuralNet *newNeuralNet, int totNeuralNet, float *total_energy, int *total_coverage, float limit, 
    int type_of_union, int random_mutation_x_block, int dim_block, float max_varaiation_mutation,cudaStream_t stream);

__global__ void recombination(NeuralNet n1, NeuralNet n2, NeuralNet final, int n_random_mutation_x_block, float max_value_mutation, int totalWeights, int totalBiases);


void wrap_neuralForward(float *mondo_cu, int* id_matrix_cu, float *mondo_signal, float *input_cu, 
                        float *output_cu, int*dim_mondo, int*cellCountMax, int*cellule_cu, float* allParams_cu,
                        int totWB, int number_of_creature);