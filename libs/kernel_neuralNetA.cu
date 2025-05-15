















void argsort_bubble(float *vettore, int *indice, int n) {
    for (int i = 0; i < n; i++) indice[i] = i;

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (vettore[indice[j]] < vettore[indice[j + 1]]) {
                int temp = indice[j];
                indice[j] = indice[j + 1];
                indice[j + 1] = temp;
            }
        }
    }
}

void argsort_bubble(int *vettore, int *indice, int n) {
    for (int i = 0; i < n; i++) indice[i] = i;

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (vettore[indice[j]] < vettore[indice[j + 1]]) {
                int temp = indice[j];
                indice[j] = indice[j + 1];
                indice[j + 1] = temp;
            }
        }
    }
}



// DA MODIFICARE / RIVEDERE SECONDO LE ULTIME MODIFICHE 
void wrapper_recombination(NeuralNet *neuralNets, NeuralNet *newNeuralNet, int totNeuralNet, float *total_energy, int *total_coverage, float limit, 
                            int type_of_union, int random_mutation_x_block, int dim_block, float max_varaiation_mutation,cudaStream_t stream){
    if(limit>1 || limit<=0) throw ("limite non valido per la generazione delle nuove reti");
    int net_order_by_energy[totNeuralNet];
    int net_order_by_coverage[totNeuralNet];

    argsort_bubble(total_energy,net_order_by_energy,totNeuralNet);
    argsort_bubble(total_coverage,net_order_by_coverage,totNeuralNet);

    int n_neuralNet = totNeuralNet*limit;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, n_neuralNet);

    int n_weight = neuralNets[0].totalWeights;
    int n_bias = neuralNets[0].totalBiases;

    for(int i=0; i<n_neuralNet; i++){
        int gen1 = distrib(gen);
        int gen2 = distrib(gen);

        int n_thread = dim_block;
        int n_block = n_weight / dim_block +1;

        recombination<<<n_block,n_thread,(random_mutation_x_block*2+1)*sizeof(int),stream>>>(neuralNets[gen1],neuralNets[gen2],newNeuralNet[i],random_mutation_x_block,max_varaiation_mutation,n_weight,n_bias);
        if(cudaGetLastError()!=cudaError::cudaSuccess) printf("recombination: %s\n",cudaGetErrorString(cudaGetLastError()));
    }

    printf("End of recombinations");


}

//Funzione per generare un nuovo modello dato i primi 2, è neccessario allocare una shared memory pari a n_random_mutation_x_block x 2
__global__ void recombination(NeuralNet n1, NeuralNet n2, NeuralNet final, int n_random_mutation_x_block, float max_value_mutation, int totalWeights, int totalBiases){

    extern __shared__ int shared_data[];

    curandState state;
    curand_init(1, threadIdx.x + blockDim.x * blockIdx.x, 0, &state);
    int random_number=-1;

    //thread 0 si occupa di copiare i dati essenziali sulla rete finale, lungheze vettori
    //inizializza anche gene predominante del blocco di pesi e bias, 0 n1, 1 n2.
    if(threadIdx.x==0){
        random_number = curand(&state) % 2;
        shared_data[0] = random_number;
        //final.totalBiases = n1.totalBiases;
        //final.totalWeights = n1.totalWeights;
        //final.numLayers = n1.numLayers;
    }

    __syncthreads();

    //I primi n thread si occupano di generare gli indici delle mutazioni dentro al blocco
    if(threadIdx.x<n_random_mutation_x_block){
        random_number = curand(&state) % blockDim.x;
        shared_data[threadIdx.x+1] = random_number;
        random_number = curand(&state) % blockDim.x;
        shared_data[threadIdx.x+1+n_random_mutation_x_block] = random_number;
    }

    __syncthreads();

    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    // Copia dei pesi secondo il gene predominante
    if(idx<totalWeights){
        if(shared_data[0]==0){
            final.d_weights[idx] = n1.d_weights[idx];
        }else{
            final.d_weights[idx] = n2.d_weights[idx];
        }
    }

    // Copia dei bias secondo il gene predominante
    if(idx<totalBiases){
        if(shared_data[0]==0){
            final.d_biases[idx] = n1.d_biases[idx];
        }else{
            final.d_biases[idx] = n2.d_biases[idx];
        }
    }
    
    // i primi nthread recuperano cio che è stato scritto in precedenza sulla shared memory 
    // prendono gli indici delle zone da mutare e li mutano in un range tra -1 e 1 per il max_value_mutation
    // si usa atomicAdd anche se peggiora le prestazioni, poiche è possibile che avvengano piu mutazioni su un peso/bias
    if(threadIdx.x<n_random_mutation_x_block){
        int idx_weights = shared_data[threadIdx.x + 1];
        int idx_bias = shared_data[threadIdx.x + 1 + n_random_mutation_x_block];

        float random_value_mutation = curand_uniform(&state) * max_value_mutation * 2 -1;
        if(idx_weights<totalWeights){
            atomicAdd(&final.d_weights[idx_weights],random_value_mutation);
        }

        random_value_mutation = curand_uniform(&state) * max_value_mutation * 2 -1;
        if(idx_bias<totalBiases){
            atomicAdd(&final.d_biases[idx_bias],random_value_mutation);
        }

    }



}



__global__ void kernel_visione(float* mondo_cu, float* mondo_signal_cu, int dim_mondo){
    if (threadid.x > dim_visione || threadid.y > dim_visione) return;
    radius_filter = dim_finestraVisione / 2;

    int filtro_x = (threadIdx.x - radius_filter + centro_x) % dim_mondo;
    int filtro_y = (threadIdx.y - radius_filter + centro_y) % dim_mondo;
    int filtro_index = filtro_y * dim_world + filtro_x;

    int visione_index = filtro_y * dim_visione + filtro_x;
    visione[visione_index] = mondo_cu[filtro_index];
    visione[visione_index + dim_visione * dim_visione] = mondo_signal_cu[filtro_index];
}


// Funzione membro che lancia il kernel
void wrap_calcolo_visione(float *mondo_cu, float *mondo_signal, float *input_cu, float *output_cu, int*dim_mondo, int*cellCountMax, int*cellule_cu, int cellCount_cu ) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);
    int iterations;
    int thread_number = 0;
    if(cellCount_cu % cellCountMax == 0){

        iterations = cellCount_cu / cellCountMax;
    }
    else{
        iterations = (cellCount_cu / cellCountMax) + 1;
    }

    for (i = 0; i < iterations; i++){
        if(i = iterations - 1): 
        {
            thread_number = cellCount_cu - cellCountMax  * iterations
        }else:
            thread_number = cellCountMax;
    }
    //computation number of thread and block for launch kernel (use max thread for dimension before launch new block)
    int n_thread_per_block = properties.maxThreadsPerBlock;
    int thread_per_dimension = sqrt(n_thread_per_block);
    int n_block = dim_world / thread_per_dimension;
    if(n_block%thread_per_dimension!=0 || n_block==0) n_block=n_block+1

    dim3 thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    dim3 block_number = dim3(n_block, n_block); 
    kernel_visione<<<block_number,thread_number,0,stream>>>(mondo_cu, mondo_signal_cu, dim_mondo);
    cudaDeviceSynchronize();  // opzionale per il debug
}



