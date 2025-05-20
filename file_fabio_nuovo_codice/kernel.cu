#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>


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

__global__ void add_creatures( float *world_value, int *world_id, int dim_world, int n_creature
                                  ) {
                                    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid <= n_creature){

        curandState state;
        curand_init(clock64(), tid, 0, &state);

        int max_attempts = 1000;
        for (int i = 0; i < max_attempts; i++) {
            int row = curand(&state) % dim_world;
            int col = curand(&state) % dim_world;
            int idx = row * dim_world + col;

            // Tentativo di occupare la cella atomica
            int old = atomicCAS(&world_id[idx], 0, tid+1);
            if (old == 0) {
                world_value[idx] = 1.0f;
                return;
            }
        }
    }

}


// ============================================================================


__global__ void fill_random_kernel(float* d_vec, int n, float minVal, float maxVal, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Setup generator per thread
    curandState state;
    curand_init(seed, idx, 0, &state);

    float rand_uniform = curand_uniform(&state); // [0,1)
    d_vec[idx] = minVal + rand_uniform * (maxVal - minVal);
}

// =========================================================================================================

template <typename T>
__global__ void resetKernel(T* d_vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_vec[idx] = T(0);  // assegna zero del tipo T
}


// ============================================================================



__global__ void recombine_models_kernel(
    float *weights, float *biases,
    float *new_weights, float *new_biases,
    int num_weights_per_model, int num_bias_per_model,
    int model1_idx, int model2_idx, int output_idx,
    float mutation_prob,
    float mutation_range,
    unsigned long seed)
{
    __shared__ int gen_id;

    // Calcolo indice thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_genes = num_weights_per_model + num_bias_per_model;
    if (idx >= total_genes) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    if(threadIdx.x==0){
        int gen = curand(&state) % 2;
        gen_id = gen==0?model1_idx:model2_idx;
    }

    __syncthreads();

    int idx_model_param_gen = -1;

    if (idx < num_weights_per_model) {

        idx_model_param_gen = (gen_id * num_weights_per_model) + idx;

        float gene_value = weights[idx_model_param_gen];

        if (curand_uniform(&state) < mutation_prob) {
            float delta = (curand_uniform(&state) * 2.0f - 1.0f) * mutation_range;
            gene_value += delta;
        }

        int idx_model_param_out = (output_idx * num_weights_per_model + idx);
        new_weights[idx_model_param_out] = gene_value;

    }else{

        idx_model_param_gen = (gen_id * num_bias_per_model) + idx - num_weights_per_model;

        float gene_value = biases[idx_model_param_gen];

        if (curand_uniform(&state) < mutation_prob) {
            float delta = (curand_uniform(&state) * 2.0f - 1.0f) * mutation_range;
            gene_value += delta;
        }

        int idx_model_param_out = (output_idx * num_bias_per_model) + idx - num_weights_per_model;
        new_biases[idx_model_param_out] = gene_value;

    }

}


// ============================================================================


__global__ void compute_energy_and_occupation_kernel(
    float* world_value,
    int* world_id,
    int* occupation_vector,
    float* energy_vector,
    int world_dim,
    int n_creature) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= world_dim || y >= world_dim) return;

    int idx = x + y*world_dim;
    
    // index on vectors valuation
    int id = world_id[idx] -1;

    if (id >= 0 && id < n_creature) {
        // Accesso atomico per evitare race condition
        atomicAdd(&occupation_vector[id], 1);
        atomicAdd(&energy_vector[id], world_value[idx]);
    }

}


// ============================================================================

__global__ void find_index_cell_alive_kernel(int *world_id, int *cell_alive_vector, int world_dim_tot, int *n_cell_alive){

    extern __shared__ int shared_mem[];
    // dim n_thread

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    shared_mem[threadIdx.x] = 0;

    if(idx==0){
        *n_cell_alive=0;
    }

    __syncthreads();

    if(idx<world_dim_tot){
        int valid_cell = world_id[idx]>0;
        cell_alive_vector[idx] = valid_cell*idx;
        shared_mem[threadIdx.x] = valid_cell;
    }

    __syncthreads();

    int n_thread_active = blockDim.x/2 +1;
    while (n_thread_active>0)
    {   
        int valid = threadIdx.x+n_thread_active<blockDim.x;
        if(threadIdx.x<n_thread_active) {
            shared_mem[threadIdx.x] = shared_mem[threadIdx.x+n_thread_active]*valid + shared_mem[threadIdx.x];
        }
        n_thread_active/=2;
        __syncthreads();
    }

    if(threadIdx.x==0){
        atomicAdd(n_cell_alive,shared_mem[0]);
    }
    
}

__global__ void compact_cell_alive_kernel_pt1(int *alive_cell_vector, int *support_vector, int *n_alive_cell, int n_block){
    extern __shared__ int shared_mem[]; 
    // dim n_thread*2+1  
    // last memory cell have the local_count of the block
    // first part the element not compacted (<blockdim.x)
    // the second part the element compacted (>blockdim.x)

    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx==0){
        shared_mem[blockDim.x*2] = 0;
    }

    //printf("----------------------alive Cell numbers: %d",n_alive_cell);

    if(idx<*n_alive_cell){
        shared_mem[threadIdx.x] = alive_cell_vector[idx];
        shared_mem[threadIdx.x+blockDim.x] = 0;
    }

    __syncthreads();

    if(threadIdx.x==0){
        int local_count = 0;
        for(int i=0; i<blockDim.x;i ++){
            if(shared_mem[i]>0){
                shared_mem[blockDim.x+local_count] = shared_mem[i];
                local_count++;
            }
        }
        shared_mem[blockDim.x*2] = local_count;
        support_vector[blockIdx.x] = local_count;
    }

    __syncthreads();

    if(threadIdx.x<shared_mem[blockDim.x*2]){
        alive_cell_vector[idx] = shared_mem[threadIdx.x+blockDim.x];
    }

}


__global__ void compact_cell_alive_kernel_pt2(int *alive_cell_vector, int *support_vector, int *n_alive_cell, int n_block, int n_thread){

    __shared__ int shared_mem[2];       
    //mem[0] starting index, mem[1] number of element 

    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(n_block==0 && threadIdx.x==0){
        shared_mem[0] = 0;
        shared_mem[1] = support_vector[n_block];
        support_vector[n_block] = shared_mem[0] + shared_mem[1];
    }else if(n_block>0 && threadIdx.x==0){
        shared_mem[0] = support_vector[n_block-1];
        shared_mem[1] = support_vector[n_block];
        support_vector[n_block] = shared_mem[0] + shared_mem[1];
    }

    __syncthreads();

    int idx_alive_cell_read = n_block*n_thread+idx;
    if(idx<shared_mem[1] && idx_alive_cell_read<*n_alive_cell){
        int offset = shared_mem[0];
        int idx_alive_cell_write = offset+idx;
        alive_cell_vector[idx_alive_cell_write] = alive_cell_vector[idx_alive_cell_read];
    }


}

// ============================================================================



__global__ void vision_kernel(
    float* world_value,
    int* world_id,
    float* world_signaling,
    int dim_world,
    int* cell_idx,
    int raggio,
    int input_workspace,
    float* input) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x; // colonna nella finestra raggio x raggio
    int y = threadIdx.y + blockIdx.y * blockDim.y; // riga nella finestra raggio x raggio

    if (x >= raggio || y >= raggio) return;

    int center_row = *cell_idx / dim_world;
    int center_col = *cell_idx % dim_world;
    printf("%d\n",*cell_idx);

    // Calcola coordinate "virtuali" senza wrapping
    int world_row = (center_row - (raggio / 2) + y + dim_world) % dim_world;
    int world_col = (center_col - (raggio / 2) + x + dim_world) % dim_world;

    int offset = (y * raggio + x) * 3;
    int workspace_offset = input_workspace*raggio*raggio*3*input_workspace;

    int world_pos = world_row * dim_world + world_col;
    input[workspace_offset + offset + 0] = world_value[world_pos];
    input[workspace_offset + offset + 1] = static_cast<float>(world_id[world_pos]);
    input[workspace_offset + offset + 2] = world_signaling[world_pos];
}

































template <typename T>
void launch_reset_kernel(T* d_vec, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    resetKernel<T><<<blocks, threads, 0, stream>>>(d_vec, n);
}


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

// Wrapper: aggiunta creature con calcolo blocchi/thread
void launch_add_creatures(float* world_value_d, int* world_id_d, int dim_world, int n_creature,
                            cudaStream_t stream) {

    int threads = n_creature;
    if(n_creature>1024) n_creature = 1024;
    int blocks = (n_creature + threads - 1) / threads;

    add_creatures<<<blocks, threads, 0, stream>>>(
        world_value_d, world_id_d, dim_world, n_creature
    );

}

// Wrapper: fill vettore random con calcolo griglia/thread
void launch_fill_random_kernel(float* d_vec, int n, float minVal, float maxVal,
                                unsigned long seed, cudaStream_t stream) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    fill_random_kernel<<<blocks, threads, 0, stream>>>(d_vec, n, minVal, maxVal, seed);

}

// Wrapper: recombine_model
void launch_recombine_models_kernel(
    float *d_weights, float *d_biases,
    float *d_new_weights, float *d_new_biases,
    int num_weights_per_model, int num_bias_per_model,
    int model1_idx, int model2_idx, int output_idx,
    float gen_x_block,
    float mutation_prob,
    float mutation_range,
    unsigned long seed,
    cudaStream_t stream) 
{
    // Numero totale di geni (pesi + bias)
    int total_genes = num_weights_per_model + num_bias_per_model;

    // Imposta configurazione kernel
    int threads_per_block = gen_x_block*total_genes +1;
    if(threads_per_block>1024) threads_per_block = 1024;
    int num_blocks = (total_genes + threads_per_block - 1) / threads_per_block;

    // Lancia il kernel
    recombine_models_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_weights,
        d_biases,
        d_new_weights,
        d_new_biases,
        num_weights_per_model,
        num_bias_per_model,
        model1_idx,
        model2_idx,
        output_idx,
        mutation_prob,
        mutation_range,
        seed
    );

}


// Wrapper compute energy and occupation for evaluation
void launch_compute_energy_and_occupation(
    float* world_value,
    int* world_id,
    int* occupation_vector,
    float* energy_vector,
    int world_dim,
    int n_creature,
    cudaStream_t stream
){

    int n_thread = 32;
    int n_block = (world_dim + n_thread -1) / n_thread;
    dim3 dim_block(n_thread,n_thread);
    dim3 dim_grid(n_block,n_block);

    compute_energy_and_occupation_kernel<<<dim_grid,dim_block,0,stream>>>(
        world_value,
        world_id,
        occupation_vector,
        energy_vector,
        world_dim,
        n_creature);


}


//Wrapper compute alive cell
void launch_find_index_cell_alive(
    int *world_id,
    int world_dim_tot,
    int *alive_cell_vector,
    int *n_cell_alive_d,
    int *support_vector,
    cudaStream_t stream
) {
    int n_thread = 1024;
    if(world_dim_tot<n_thread) n_thread = world_dim_tot;
    if(n_thread%2==1) n_thread++;
    int n_block = (world_dim_tot+n_thread-1) / n_thread;

    find_index_cell_alive_kernel<<<n_block,n_thread,sizeof(int)*n_thread,stream>>>(
        world_id,
        alive_cell_vector,
        world_dim_tot,
        n_cell_alive_d
    );

    int n;
    cudaMemcpy(&n,n_cell_alive_d,sizeof(int),cudaMemcpyDeviceToHost);
    printf("================================ n: %d ============================\n",n);

    n_thread = 32;
    n_block = (world_dim_tot+n_thread-1) / n_thread;

    compact_cell_alive_kernel_pt1<<<n_block,n_thread,sizeof(int)*(n_thread*2+1),stream>>>(
        alive_cell_vector,
        support_vector,
        n_cell_alive_d,
        n_block
    );

    for(int i=0; i<n_block; i++){
        compact_cell_alive_kernel_pt2<<<1,n_thread,0,stream>>>(
            alive_cell_vector,
            support_vector,
            n_cell_alive_d,
            i,
            n_thread
        );
    }

    


}



// Wrapper kernel visione
void launch_vision(    
    float* world_value,
    int* world_id,
    float* world_signaling,
    int dim_world,
    int* cell_idx,
    int raggio,
    int input_workspace,
    float* input,
    cudaStream_t stream
){

    int n_thread = 32;
    int n_block = (raggio + n_thread -1) / n_thread;
    dim3 threads(n_thread,n_thread);
    dim3 blocks(n_block,n_block);
    vision_kernel<<<blocks,threads,0,stream>>>(world_value,world_id,world_signaling,dim_world,cell_idx,raggio,input_workspace,input);


}