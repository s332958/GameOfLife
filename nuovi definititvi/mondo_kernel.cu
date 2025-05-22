// =========================================================================================================

__global__ void add_objects_to_world(float *world_value, int *world_id, int dim_world, 
                                    int id, float min_value, float max_value, float threashold
                                ){
    
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int idx = x + y*dim_world;

    if(idx<dim_world*dim_world){

        if(world_id[idx]==0){
            curandState state;
            curand_init(clock64(),threadIdx.x,0,&state);
            float p_occupation = curand_uniform(&state);

            if(p_occupation>threashold){
                float value = curand_uniform(&state)*(max_value - min_value) + (min_value);
                world_id[idx] = id;
                world_value[idx] = value;
            }

        }
    
    }

}

// =========================================================================================================

_global__ void world_update_kernel(float *mondo_creature, float *world, int *id_matrix, 
                                int dim_world, int number_of_creatures, int *cellCount, Cellula *cellule_cu, int convolution_iter)
    {                        
    int center_x = blockIdx.x*blockDim.x + threadIdx.x;    
    int center_y = blockIdx.y*blockDim.y + threadIdx.y;
    int index = center_y * dim_world + center_x;
    
    if(index >= dim_world*dim_world) return;
    
    
    int ID = id_matrix[index];
    float starting_value = world[index];
    
    if(ID == -1) return;

    float final_value = starting_value;
    int final_id = ID;

    int max_id = 0;
    int max_value = 0;
    float ally_energy = 0;
    float enemy_energy = 0;
    for(int i = 1; i <= number_of_creatures; i++){
        int value = mondo_creature[i * dim_world * dim_world + index]; 
        if (i == ID){
            ally_energy = value;      
        }    
        else{
            enemy_energy += value;
        }                      
        if (value > max_value){
            max_value = value;
            max_id = i;
        }    
    }    
    if (ID == 0){
        if (enemy_energy > 0){
            final_value = starting_value * (max_value / enemy_energy) + max_value;
            final_id = max_id;

            pos = atomicAdd(cellCount, 1);
            Cellula newCell;
            nuova.index = index;
            nuova.ID = final_id;
            nuova.alive = true;
            nuova.dim_visione = 9; //dim_visione
            nuova.center_x = index % dim_mondo;
            nuova.center_y = index / dim_mondo;

            cellule_cu[pos] = newCell;
        }    
    }    
    else{
        if (starting_value + ally_energy - enemy_energy < 0){
            final_value = abs(starting_value + ally_energy - enemy_energy);
            final_id = 0;
        }            
        else{
            final_value = starting_value + ally_energy - enemy_energy;
        }    
    }    

    /*
    final_value = fmaxf(0.0, fminf(255.0, final_value)); 
    int index_save = convolution_iter*dim_world*dim_world + index;
    world_save[index_save] = (unsigned char)final_value;
    id_matrix_save[index_save] = (unsigned char)final_id;
    */ 
   if(final_value < 20.0f){
        final_id = 0;
   }     

   if(final_value > 255.0f){
        final_value = 255.0f;
    }    

    world[index] = final_value;                   
    id_matrix[index] = (int)final_id; 


}    

//==============================================================================================

__global__ void cellule_cleanup_kernel(Cellula *cellule_cu, int *id_matrix, int* cellCount, int* mask_cu){
    __shared__ int dim_paral = 10;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx > cellCount) return;
    
    if(id_matrix[cellule_cu[idx].index] <= 0){
       cellule_cu[idx].alive = false; 
    }    

    mask_cu[idx] = cellule_cu[idx].alive;
    id_sort_x = idx % dim_paral;
    id_sort_y = idx / dim_paral;
    
    if (id_sort_x == 0){
        int increment = 0;
        for (int i = 0; i < dim_paral; i++){
            if(id_sort_y + i < cellCount){
                increment = increment + mask_cu[id_sort_y*dim_paral + i];
                mask_cu[id_sort_y*dim_paral + i] = increment;
            }                
        }    
    }    
    __syncthreads();

    if (id_sort_y != 0){
        mask_cu[id_sort_y*dim_paral + id_sort_x] = mask_cu[id_sort_y*dim_paral + id_sort_x] + mask_cu[id_sort_y*dim_paral - 1];
    }    
    __syncthreads();

    if (cellule[idx].alive){
        cellule[mask[idx] - 1] = cellule[idx];
    }    

    if (idx != 0) return;
    cellCount = mask_cu[cellCount - 1];
} 

//================================================================================

//Wrapper add objects to world
void launch_add_objects_to_world(float* world_value_d, int* world_id_d, int dim_world,
                                int id, float min_value, float max_value, float threshold,
                                cudaStream_t stream) {

    dim3 blockDim(16, 16);
    dim3 gridDim((dim_world + 15) / 16, (dim_world + 15) / 16);

    add_objects_to_world<<<gridDim, blockDim, 0, stream>>>(
        world_value_d, world_id_d, dim_world,
        id, min_value, max_value, threshold
    );

}

//Wrapper mondo_cu update
void launch_world_update(float *mondo_creature, float *world, int *id_matrix, int dim_world, int number_of_creatures,
                                     int *cellCount, Cellula *cellule_cu, int convolution_iter);{


    n_thread_per_block = properties.maxThreadsPerBlock;  
    int thread_per_dimension = sqrt(n_thread_per_block);
    n_block = dim_world / thread_per_dimension;
    if(n_block==0) n_block = 1;
    thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    block_number = dim3(n_block, n_block); 

    world_update_kernel<<<block_number,thread_number,0,stream>>>(mondo_creature, world,id_matrix,filter,world_save,id_matrix_save,dim_world,number_of_creatures,convolution_iter);

}

//Wrapper cellule cleanup
void launch_cellule_cleanup(Cellula *cellule_cu, int* cellCount, int* mask_cu){
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    int n_thread_per_block = properties.maxThreadsPerBlock;  
    int thread_per_dimension = sqrt(n_thread_per_block);
    int n_block = sqrt(cellCount) / thread_per_dimension;
    if(n_block==0) n_block++;
    dim3 block = dim3(n_block,n_block);
    dim3 thread = dim3(thread_per_dimension,thread_per_dimension);

    cellule_cleanup_kernel<<<block,thread>>>(cellule_cu, cellCount, mask_cu);
}