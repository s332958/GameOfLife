#include "libs/Cellula.cuh"
#include "libs/neural_net.cuh"

//inizializzazione
const int dim_mondo = WIDTH;
const int MaxPixel = dim_mondo * dim_mondo;
const float memoriaGB = 2;
const int numero_stream = 1;
const int number_of_creatures = 10;
//const int Races = 10;
//const int SubRaces = 10;
//number_of_creatures = SubRaces * Races;

int cellule[MaxPixel];
int cellCount = 0;

NeuralNet NNmodels [number_of_creatures]; 
int dim_inputs = 162;
int dim_outputs = 10;

int layers [] = {dim_inputs, 16, 16, dim_outputs};
int numLayers = layers.size();
int totW = 0;
int totB = 0; 
int totWB = 0;



//calcolo numero di parametri:
for (int i = 1; i < numLayers; ++i) {
    totW += sizes[i - 1] * sizes[i]; //weights
    totB += sizes[i]; //biases
}
totWB = totW + totB;
//riempimento di array cellule:
for(int i = 0; i < MaxPixel; i++){
    if(mondo[i] > 0){
        cellule[cellCount] = i;//assegnato ad una razza e sottorazza
        cellCount = cellCount + 1;
    }    
}
//riempimento iniziale di array NNmodels:
//idealmente razze definite da valori random e sottorazze definite da piccoli incrementi o decrementi dei valori random
float params [totWB];
int dim_allparams = totWB * number_of_creatures;
float allparams [dim_allparams];
int value;

for (int i = 0; i < number_of_creatures; i++){
    for (int i = 0; i < totWB; i++){
        value = rand();
        params [i] = value;
        allparams [dim_allparams] = value;
        dim_allparams = dim_allparams + 1;
    }  
    //vedi classe NeuralNet in neural_net.h e neural_net.c++
    NNmodels [i] = new NeuralNet (layers, numLayers, params, totW, totB);
}


/*
for (int i = 0; i < Races; i++){
    for (int j = 0; j < SubRaces; j++){
        if (j==0){
            for (int i = 0; i < totWB; i++){
                params [i] = rand();
            }            
        }
        else{
            for (int i = 0; i < totWB; i++){
                params [i] = params [i] + increment*((rand()*2)-1);
            }  
        }
    }    
    NNmodels [i] = new NeuralNet (layers, numLayers, params);
}
*/

//calcolo numero massimo di cellule di cui salvare input e output per non sforare la memoria
cellCountMax = (memoriaGB * 1073741824 - dim_allparams * 4)/(dim_inputs + dim_outputs) * 4;   
cellCountMax = cellCountMax / numero_stream;

if(cellCountMax >= cellCount){
    cellCountMax = cellCount;
}

int *cellCount_cu, *mask_cu, *cellule_cu;
if(compute_capability>=7) controllo_errore_cuda("allocazione cellCount_cu", cudaMallocAsync((void**)&cellCount_cu, sizeof(int),stream));
else controllo_errore_cuda("allocazione cellCount_cu", cudaMalloc((void**)&cellCount_cu, sizeof(int)));
if(compute_capability>=7) controllo_errore_cuda("allocazione mask_cu", cudaMallocAsync((void**)&mask_cu, MaxPixel*sizeof(int),stream));
else controllo_errore_cuda("allocazione mask_cu", cudaMalloc((void**)&mask_cu, MaxPixel*sizeof(int)));
if(compute_capability>=7) controllo_errore_cuda("allocazione cellule_cu", cudaMallocAsync((void**)&cellule_cu, MaxPixel*sizeof(int),stream));
else controllo_errore_cuda("allocazione cellule_cu", cudaMalloc((void**)&cellule_cu, MaxPixel*sizeof(int)));
/*
da integrare a questi del main.cpp
float *mondo_cu, *mondo_creature_cu;
int *id_matrix_cu;
*/

float *allParams_cu, *NNmodels_evaluation_cu, *input_cu, *output_cu, *mondo_signal_cu;
//mondo_signal_cu
if(compute_capability>=7) controllo_errore_cuda("allocazione mondo", cudaMallocAsync((void**)&mondo_signal_cu, dim_mondo*dim_mondo*sizeof(float),stream));
else controllo_errore_cuda("allocazione mondo", cudaMalloc((void**)&mondo_signal_cu, dim_mondo*dim_mondo*sizeof(float)));
//allParams_cu
if(compute_capability>=7) controllo_errore_cuda("allocazione allParams_cu", cudaMallocAsync((void**)&allParams_cu, dim_allparams*number_of_creatures*sizeof(float),stream));
else controllo_errore_cuda("allocazione allParams_cu", cudaMalloc((void**)&allParams_cu, dim_allparams*number_of_creatures*sizeof(float)));
//NNmodels_evaluation_cu
if(compute_capability>=7) controllo_errore_cuda("allocazione NNmodels_evaluation_cu", cudaMallocAsync((void**)&NNmodels_evaluation_cu, number_of_creatures*sizeof(float),stream));
else controllo_errore_cuda("allocazione NNmodels_evaluation_cu", cudaMalloc((void**)&NNmodels_evaluation_cu, number_of_creatures*sizeof(float)));
//input_cu grande come cellCountMax*dim_inputs*sizeof(float)
if(compute_capability>=7) controllo_errore_cuda("allocazione input_cu", cudaMallocAsync((void**)&input_cu, cellCountMax*dim_inputs*sizeof(float),stream));
else controllo_errore_cuda("allocazione input_cu", cudaMalloc((void**)&input_cu, cellCountMax*dim_inputs*sizeof(float)));
//output_cu grande come cellCountMax*dim_outputs*sizeof(float)
if(compute_capability>=7) controllo_errore_cuda("allocazione *output_cu", cudaMallocAsync((void**)&output_cu, cellCountMax*dim_outputs*sizeof(float),stream));
else controllo_errore_cuda("allocazione *output_cu", cudaMalloc((void**)&output_cu, cellCountMax*dim_outputs*sizeof(float)));

controllo_errore_cuda("passaggio cellule_cu su GPU", cudaMemcpyAsync(allParams_cu, allparams, dim_allparams*sizeof(float), cudaMemcpyHostToDevice, stream));
controllo_errore_cuda("passaggio cellule_cu su GPU", cudaMemcpyAsync(cellule_cu, cellule, MaxPixel*sizeof(int), cudaMemcpyHostToDevice, stream));
controllo_errore_cuda("passaggio cellCount_cu su GPU", cudaMemcpyAsync(cellCount_cu, cellCount, sizeof(int), cudaMemcpyHostToDevice, stream));

//simulazione:
for (int k = 0; k < NSimulazioni, k++){

    for (int j = 0; j < MaxSimulazione; j++){
        //in kernel.cu
        wrap_zero_mondocreature(mondo_creature, dim_world);

        //in kernel_neuralNetA
        wrap_neuralForward(mondo_cu, mondo_signal, input_cu, dim_inputs, output_cu, dim_outputs, dim_mondo, cellCountMax, cellule_cu, cellCount_cu, allParams_cu, totWB, number_of_creatures); 

        //in kernel.cu
        wrap_mondo_cu_update(float *mondo_creature, float *world, int *id_matrix, int dim_world, int number_of_creatures,
                             int *cellCount, Cellula *cellule_cu, int convolution_iter)//vivo morto chi vince (regole del mondo) da aggiustare anche lui

        //(in kernel.cu) riscrive cellule_cu eliminando le cellule morte compattando inoltre l'array e aggiorna cellCount
        wrap_cellule_cleanup(cellule_cu, id_matrix_cu, cellCount_cu, mask_cu);//da aggiustare per il cambio di struttura
    }
    //(in kernel.cu) salva la sommatoria dei valori delle creature negli indici dei NNmodels corrispettivi
    wrap_evaluation(cellule_cu, cellCount, NNmodels_evaluation_cu);

    //(in kernel.cu) trova i primi di NNmodels_evaluation_cu per razza e aggiorna l'array NNmodels con i nuovi modelli ottenuti dalle ricombinazioni
    wrapper_recombination();
}









