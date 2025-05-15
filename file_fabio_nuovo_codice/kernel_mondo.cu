#include <cuda_runtime.h>
#include <curand_kernel.h>


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



__global__ void place_creatures( float *world_value, int *world_id, int dim_world, int n_creature
                                  ) {
                                    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid <= n_creature){

        curandState state;
        curand_init(clock64(), tid, 0, &state);

        int max_attempts = 1000;
        for (int i = 0; i < max_attempts; i) {
            int row = curand(&state) % dim_world;
            int col = curand(&state) % dim_world;
            int idx = row * dim_world + col;

            // Tentativo di occupare la cella atomica
            int old = atomicCAS(&world_id[idx], 0, 1);
            if (old == 0) {
                world_value[idx] = 1.0f;
                return;
            }
        }
    }

}


// ============================================================================

__global__ void set_to_zero(float *matrix_contribution, int world_dim, int number_creature){

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;

    int idx = x + y*world_dim + z*world_dim*world_dim;

    if(idx<world_dim*world_dim*number_creature){
        matrix_contribution[idx]=0;
    }

}