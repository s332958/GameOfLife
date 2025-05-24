
__global__ void mappa_colori_kernel(float* mondo, int* id_matrix, float* mondo_rgb, int thread_number) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= thread_number) return;

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
        mondo_rgb[center_index + RGB_index] = colori[ID][RGB_index]*mondo[center_index]/65025.0f;
    } else {
        mondo_rgb[center_index + RGB_index] = 0.0f; 
    }
}


void launch_mappa_colori(float* mondo, int* id_matrix, float* mondo_rgb_d, int world_dim){

    int n_thread_per_block = 1024; //properties.maxThreadsPerBlock; 
    int thread_number = world_dim*world_dim*3;
    int n_block = thread_number / n_thread_per_block;
    if(n_block%n_thread_per_block!=0 || n_block==0)n_block=n_block+1; 

    mappa_colori_kernel<<<n_block, n_thread_per_block, 0, stream>>>(mondo, id_matrix, mondo_rgb_d, thread_number);

}