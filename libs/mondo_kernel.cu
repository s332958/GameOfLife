#include <cuda_runtime.h>
#include <curand_kernel.h>

// =========================================================================================================

__global__ void add_objects_to_world_kernel(float *world_value, int *world_id, int dim_world, 
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

__global__ void world_update_kernel(
    float *world_value, 
    int *id_matrix, 
    float *contribution_matrix,
    int dim_world, 
    int number_of_creatures, 
    int *cellCount, 
    int *cells
)
    {                        
        int center_x = blockIdx.x*blockDim.x + threadIdx.x;    
        int center_y = blockIdx.y*blockDim.y + threadIdx.y;
        int index = center_y * dim_world + center_x;
        
        if(index >= dim_world*dim_world) return;
        
        
        int ID = id_matrix[index];
        float starting_value = world_value[index];
        
        if(ID == -1) return;
        
        float final_value = starting_value;
        int final_id = ID;
        
        int max_id = 0;
        int max_value = 0;
        float ally_energy = 0;
        float enemy_energy = 0;
        for(int i = 0; i < number_of_creatures; i++){
            int value = contribution_matrix[i * dim_world * dim_world + index]; 
            if (i+1 == ID){
                ally_energy = value;      
            }    
            else{
                enemy_energy += value;
            }                      
            if (value > max_value){
                max_value = value;
                max_id = i+1;
            }    
        }    
        if (ID == 0){
            if (enemy_energy > 0){
                final_value = starting_value * (max_value / enemy_energy) + max_value;
                final_id = max_id;
                
                int pos = atomicAdd(cellCount, 1);
                cells[pos] = index;
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
        
        if(final_value < 0.2f){
           final_id = 0;
        }     
        
        if(final_value > 1.0f){
            final_value = 1.0f;
        }    
        
        world_value[index] = final_value;                   
        id_matrix[index] = (int)final_id; 
    }    
    
    //==============================================================================================
    
__global__ void cellule_cleanup_kernel(int *cellule_cu, int* temp_cellule_cu, int *id_matrix, int* cellCount, int* mask_cu, bool* mask_alive){
        
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        //ritorno i thread che eccedono il numero di cellule vive
        if(idx >= *cellCount) return;

        //inizializzo la maschera a 0 e temp cellule a 0 e alive a false
        mask_cu[idx] = 0;
        temp_cellule_cu[idx] = 0;
        mask_alive[idx] = false;
    
        // se trovo cellule con id > 0 quindi occupate da creature segno 1 in mask_cu e true in mask_alive
        if(id_matrix[cellule_cu[idx]] > 0){
           mask_cu[idx] = 1; 
           mask_alive[idx] = true;
        }  
        // dichiaro un valore di parallelizzazione?  
        int dim_paral = 10;
        int id_sort_x = idx % dim_paral;
        int id_sort_y = idx / dim_paral;
        
        // indice di ordinamento?
        int index_sort = id_sort_y * dim_paral + id_sort_x;
        
        if (id_sort_x == 0){
            int increment = 0;
            for (int i = 0; i < dim_paral; i++){
                
                if(index_sort + i < *cellCount){ 
                    increment = increment + mask_cu[index_sort + i];               
                    mask_cu[index_sort + i] = increment;
                }
                                
            }    
        }    
        __syncthreads();
    
        if (id_sort_y != 0){
            mask_cu[index_sort] = mask_cu[index_sort] + mask_cu[id_sort_y*dim_paral - 1];
        }    
        __syncthreads();
    
        if (mask_alive[idx] == true){
            temp_cellule_cu[mask_cu[idx] - 1] = cellule_cu[idx];
        }     
        int new_cellCount = mask_cu[*cellCount - 1];
        if (idx > new_cellCount)return;
    
        cellule_cu[idx] = temp_cellule_cu[idx];  
    
        if (idx != 0) return;
        *cellCount = new_cellCount; 
    
        
    } 
    
    //================================================================================
    
    //Wrapper add objects to world
void launch_add_objects_to_world(float* world_value_d, int* world_id_d, int dim_world,
                                int id, float min_value, float max_value, float threshold,
                                cudaStream_t stream) {

    dim3 blockDim(16, 16);
    dim3 gridDim((dim_world + 15) / 16, (dim_world + 15) / 16);

    add_objects_to_world_kernel<<<gridDim, blockDim, 0, stream>>>(
        world_value_d, world_id_d, dim_world,
        id, min_value, max_value, threshold
    );

}

//Wrapper mondo_cu update
void launch_world_update(
    float *world_value,
    int *id_matrix,
    float *contribution_matrix, 
    int *cells,
    int world_dim, 
    int number_of_creatures,
    int *cellCount, 
    cudaStream_t stream
){


    int n_thread_per_block = 1024;  
    int thread_per_dimension = sqrt(n_thread_per_block);
    int n_block = world_dim / thread_per_dimension;
    if(n_block==0) n_block = 1;
    dim3 thread_number = dim3(thread_per_dimension, thread_per_dimension);  
    dim3 block_number = dim3(n_block, n_block); 

    world_update_kernel<<<block_number,thread_number,0,stream>>>(
        world_value, 
        id_matrix,
        contribution_matrix,
        world_dim,
        number_of_creatures,
        cellCount,
        cells
    );

}

//Wrapper cellule cleanup
void launch_cellule_cleanup(int* cells, int* cellCount, int* id_matrix, cudaStream_t stream){
    int* temp_cellule_cu;
    int* mask_cu;
    bool* mask_alive;    
    
    cudaMalloc((void**)&mask_alive, *cellCount * sizeof(bool));
    cudaMalloc((void**)&mask_cu, *cellCount * sizeof(int));
    cudaMalloc((void**)&temp_cellule_cu, *cellCount * sizeof(int));

    int n_thread_per_block = 1024; //properties.maxThreadsPerBlock; 
    int thread_number = *cellCount;
    int n_block = thread_number / n_thread_per_block;
    if(n_block%n_thread_per_block!=0 || n_block==0)n_block=n_block+1; 

    cellule_cleanup_kernel<<<n_block, n_thread_per_block, 0, stream>>>(cells, temp_cellule_cu, id_matrix, cellCount, mask_cu, mask_alive);

    cudaFree(temp_cellule_cu);
    cudaFree(mask_alive);
    cudaFree(mask_cu);
}
