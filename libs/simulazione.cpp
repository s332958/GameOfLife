#include "libs/Cellula.cuh"
#include "libs/neural_net.cuh"

//inizializzazione
const int dim_mondo = WIDTH;
const int MaxPixel = dim_mondo * dim_mondo;
const float memoriaGB = 2;
const int numero_stream = 1;
//const int Races = 10;
//const int SubRaces = 10;
//number_of_creatures = SubRaces * Races;

int cellule[MaxPixel];
int cellCount = 0;

NeuralNet NNmodels [number_of_creatures]; 
int dim_inputs = 162;
int dim_outputs = 10;

int layers [] = {inputs, 16, 16, dim_outputs};
int numLayers = layers.size();
int totW = 0;
int totB = 0; 
int totWB = 0;

float *mondo_signal_cu,

if(compute_capability>=7) controllo_errore_cuda("allocazione mondo", cudaMallocAsync((void**)&mondo_signal_cu, dim_mondo*dim_mondo*sizeof(float),stream));
else controllo_errore_cuda("allocazione mondo", cudaMalloc((void**)&mondo_signal_cu, dim_mondo*dim_mondo*sizeof(float)));

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

//trasferimento di cellule[] e di NNmodels[] su GPU:
cellCountMax = (memoriaGB * 1073741824 - dim_allparams * 4)/(dim_inputs + dim_outputs) * 4;   
cellCountMax = cellCountMax / numero_stream;

int *cellCount_cu, *mask_cu, *cellule_cu;
if(compute_capability>=7) controllo_errore_cuda("allocazione cellCount_cu", cudaMallocAsync((void**)&cellCount_cu, sizeof(int),stream));
else controllo_errore_cuda("allocazione cellCount_cu", cudaMalloc((void**)&cellCount_cu, sizeof(int)));
if(compute_capability>=7) controllo_errore_cuda("allocazione mask_cu", cudaMallocAsync((void**)&mask_cu, MaxPixel*sizeof(int),stream));
else controllo_errore_cuda("allocazione mask_cu", cudaMalloc((void**)&mask_cu, MaxPixel*sizeof(int)));
if(compute_capability>=7) controllo_errore_cuda("allocazione cellule_cu", cudaMallocAsync((void**)&cellule_cu, MaxPixel*sizeof(int),stream));
else controllo_errore_cuda("allocazione cellule_cu", cudaMalloc((void**)&cellule_cu, MaxPixel*sizeof(int)));



float *allParams_cu, *NNmodels_evaluation_cu, *input_cu, *output_cu;
if(compute_capability>=7) controllo_errore_cuda("allocazione allParams_cu", cudaMallocAsync((void**)&allParams_cu, dim_allparams*sizeof(float),stream));
else controllo_errore_cuda("allocazione allParams_cu", cudaMalloc((void**)&allParams_cu, dim_allparams*sizeof(float)));
if(compute_capability>=7) controllo_errore_cuda("allocazione NNmodels_evaluation_cu", cudaMallocAsync((void**)&NNmodels_evaluation_cu, number_of_creatures*sizeof(float),stream));
else controllo_errore_cuda("allocazione NNmodels_evaluation_cu", cudaMalloc((void**)&NNmodels_evaluation_cu, number_of_creatures*sizeof(float)));
if(compute_capability>=7) controllo_errore_cuda("allocazione input_cu", cudaMallocAsync((void**)&input_cu, cellCountMax*dim_inputs*sizeof(float),stream));
else controllo_errore_cuda("allocazione input_cu", cudaMalloc((void**)&input_cu, cellCountMax*dim_inputs*sizeof(float)));
if(compute_capability>=7) controllo_errore_cuda("allocazione *output_cu", cudaMallocAsync((void**)&output_cu, cellCountMax*dim_outputs*sizeof(float),stream));
else controllo_errore_cuda("allocazione *output_cu", cudaMalloc((void**)&output_cu, cellCountMax*dim_outputs*sizeof(float)));

controllo_errore_cuda("passaggio cellule_cu su GPU", cudaMemcpyAsync(allParams_cu, allparams, dim_allparams*sizeof(float), cudaMemcpyHostToDevice, stream));
controllo_errore_cuda("passaggio cellule_cu su GPU", cudaMemcpyAsync(cellule_cu, cellule, MaxPixel*sizeof(int), cudaMemcpyHostToDevice, stream));
controllo_errore_cuda("passaggio cellCount_cu su GPU", cudaMemcpyAsync(cellCount_cu, cellCount, sizeof(int), cudaMemcpyHostToDevice, stream));

//simulazione:
for (int k = 0; k < NSimulazioni, k++){

    for (int j = 0; j < MaxSimulazione; j++){
        wrap_calcolo_visione(mondo_cu, mondo_signal, input_cu, output_cu, dim_mondo, cellCountMax, cellule_cu, cellCount_cu); 
        for (int i = 0; i < cellCount; i++){        
            //calcolo valori di input:
                   
        }

        for (int i = 0; i < cellCount; i++){              
            //Forward nel NN:
            forwardOnDevice(input_cu[i*dim_inputs], output_cu[i*dim_inputs], );
        }
    
        //(in kernel.cu) modifica mondo_cu e mondo_signals_cu usando il vettore Output aggiornato in ogni cellula di cellule_cu
        //inoltre scopre le celle vive che muoiono e le cambia il valore boolean alive in false.
        //le cellule che nascono vengono aggiunte in coda a cellule_cu
        wrap_convolution(cellule_cu, mondo_creature_cu, mondo_cu, id_matrix_cu, dim_mondo, number_of_creatures, cellCount_cu, dim_output, convolution_iter, stream);
                                

        //(in kernel.cu) riscrive cellule_cu eliminando le cellule morte compattando inoltre l'array e aggiorna cellCount
        wrap_cellule_cleanup(cellule_cu, id_matrix_cu, cellCount_cu, mask_cu);
    }
    //(in kernel.cu) salva la sommatoria dei valori delle creature negli indici dei NNmodels corrispettivi
    wrap_evaluation(cellule_cu, cellCount, NNmodels_evaluation_cu);

    //(in kernel.cu) trova i primi di NNmodels_evaluation_cu per razza e aggiorna l'array NNmodels con i nuovi modelli ottenuti dalle ricombinazioni
}









