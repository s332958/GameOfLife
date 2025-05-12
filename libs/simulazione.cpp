#include "libs/Cellula.cuh"
#include "libs/neural_net.cuh"

//inizializzazione
const int dim_mondo = WIDTH;
const int MaxPixel = WIDTH * HEIGHT;
const int Races = 10;
const int SubRaces = 10;


Cellula cellule[MaxPixel];
cellCount = 0;
number_of_creatures = SubRaces * Races;
NeuralNet NNmodels [number_of_creatures]; 

int layers [] = {81, 16, 16, 10};
int numLayers = layers.size();
int totWB = 0;

//calcolo numero di parametri:
for (int i = 1; i < numLayers; ++i) {
    totWB += sizes[i - 1] * sizes[i]; //weights
    totWB += sizes[i]; //biases
    }

//riempimento di array cellule:
for(int i = 0; i < MaxPixel; i++){
    if(mondo[i] > 0){
        cellule[cellCount] = Cellula(cellCount, mondo_id[cellCount] + rand(SubRaces));//assegnato ad una razza e sottorazza
        cellCount = cellCount + 1;
    }    
}
//riempimento iniziale di array NNmodels:
//idealmente razze definite da valori random e sottorazze definite da piccoli incrementi o decrementi dei valori random
float params [totWB];
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

//trasferimento di cellule[] e di NNmodels[] su GPU:
Cellula *cellule_cu;
NeuralNet *NNmodels_cu;
float *NNmodels_evaluation_cu;

if(compute_capability>=7) controllo_errore_cuda("allocazione mondo", cudaMallocAsync((void**)&cellule_cu, MaxPixel*sizeof(Cellula),stream));
else controllo_errore_cuda("allocazione mondo", cudaMalloc((void**)&cellule_cu, MaxPixel*sizeof(Cellula)));

if(compute_capability>=7) controllo_errore_cuda("allocazione mondo", cudaMallocAsync((void**)&NNmodels_cu, number_of_creatures*sizeof(NeuralNet),stream));
else controllo_errore_cuda("allocazione mondo", cudaMalloc((void**)&NNmodels_cu, number_of_creatures*sizeof(NeuralNet)));

if(compute_capability>=7) controllo_errore_cuda("allocazione mondo", cudaMallocAsync((void**)&NNmodels_evaluation_cu, number_of_creatures*sizeof(float),stream));
else controllo_errore_cuda("allocazione mondo", cudaMalloc((void**)&NNmodels_evaluation_cu, number_of_creatures*sizeof(float)));

controllo_errore_cuda("passaggio mondo su GPU", cudaMemcpyAsync(cellule_cu, cellule, MaxPixel*sizeof(Cellula), cudaMemcpyHostToDevice, stream));
controllo_errore_cuda("passaggio id_matrix su GPU", cudaMemcpyAsync(NNmodels_cu, NNmodels, number_of_creatures*sizeof(NeuralNet), cudaMemcpyHostToDevice, stream));

//simulazione:
for (int k = 0; k < NSimulazioni, k++){

    for (int j = 0; j < MaxSimulazione; j++){
        for (int i = 0; i < cellCount; i++){
        
            //calcolo valori di input:
            cellule_cu[i].kernel_visione();
        
            //Forward nel NN:
            NNmodels_cu[cellule_cu[i].ID].forwardOnDevice(cellule_cu[i].visione, cellule_cu[i].output);
        }
    
        //(in kernel.cu) modifica mondo_cu e mondo_signals_cu usando il vettore Output aggiornato in ogni cellula di cellule_cu
        //inoltre scopre le celle vive che muoiono e le cambia il valore boolean alive in false.
        //le cellule che nascono vengono aggiunte in coda a cellule_cu
        wrap_elaboration(cellule_cu);

        //(in kernel.cu) riscrive cellule_cu eliminando le cellule morte compattando inoltre l'array e aggiorna cellCount
        wrap_cellule_cleanup(cellule_cu);
    }
    //(in kernel.cu) salva la sommatoria dei valori delle creature negli indici dei NNmodels corrispettivi
    wrap_evaluation(cellule_cu, cellCount, NNmodels_evaluation_cu);

    //(in kernel.cu) trova i primi di NNmodels_evaluation_cu per razza e aggiorna l'array NNmodels con i nuovi modelli ottenuti dalle ricombinazioni
}









