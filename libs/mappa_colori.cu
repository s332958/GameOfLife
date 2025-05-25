// Dichiaro la memoria costante in GPU dove verranno tenuti i colori per poi fare il rendering
extern __constant__ float COLORI[100][3];
// Numero di colori massimi 
const int MAX_CREATURE = 100;

// Dichiaro i colori in host
void generateDistinctColors(int* colors) {
    for (int i = 0; i < MAX_CREATURE; ++i) {
        // HSV -> RGB conversion per generare colori distinti
        float h = (i * 360.0f / MAX_CREATURE);
        float s = 1.0f;
        float v = 1.0f;

        float c = v * s;
        float x = c * (1 - fabs(fmod(h / 60.0f, 2) - 1));
        float m = v - c;

        float r, g, b;
        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        colors[i*3+0] = r+m;
        colors[i*3+1] = b+m;
        colors[i*3+2] = g+m;
    }
}

// Passo i colori in GPU e poi libero l'allocazione in Host
void load_constant_memory_GPU() {
    int *color_h = (int*) malloc(sizeof(int)*MAX_CREATURE*3);
    generateDistinctColors(color_h);

    cudaMemcpyToSymbol(COLORI, color_h, sizeof(float) * 100 * 3);
    free(color_h);
}

// Kernel per generare i colori corrispettivi dati mondo_value, monod_id, e l'output mondo_rgb
__global__ void mappa_colori_kernel(float* mondo, int* id_matrix, float* mondo_rgb, int world_dim) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= world_dim) return;

    int center_index = index / 3;
    int RGB_index = index % 3;
    
    int ID = id_matrix[center_index];
    
    if (ID == -1) {//rosso
        if(RGB_index == 0){
            mondo_rgb[center_index + RGB_index] = 255.0f;
        }else{
            mondo_rgb[center_index + RGB_index] = 0.0f;
        }
    } else if (ID == 0) {
        mondo_rgb[center_index + RGB_index] = 255.0f*mondo[center_index]/65025.0f;
    } else if (ID >= 1 && ID <= 20) {
        mondo_rgb[center_index + RGB_index] = COLORI[ID][RGB_index]*mondo[center_index]/65025.0f;
    } else {
        mondo_rgb[center_index + RGB_index] = 0.0f; 
    }
}


void launch_mappa_colori(float* mondo, int* id_matrix, float* mondo_rgb_d, int world_dim, cudaStream_t stream){

    int n_thread_per_block = 1024; //properties.maxThreadsPerBlock; 
    int thread_number = world_dim*world_dim*3;
    int n_block = (thread_number + n_thread_per_block - 1) / n_thread_per_block;


    mappa_colori_kernel<<<n_block, n_thread_per_block, 0, stream>>>(mondo, id_matrix, mondo_rgb_d, thread_number);

}