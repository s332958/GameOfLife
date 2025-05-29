// mondo_kernel.cuh
#pragma once

#include <cuda_runtime.h>

// Wrapper per aggiungere oggetti al mondo
void launch_add_objects_to_world(float* world_value_d, int* world_id_d, int dim_world,
                                 int id, float min_value, float max_value, float threshold,
                                 cudaStream_t stream);

// Wrapper per aggiornare lo stato del mondo
void launch_world_update(float* world_value,
                         int* id_matrix,
                         float *world_signal,
                         float* contribution_matrix, 
                         int* cells,
                         int world_dim, 
                         int number_of_creatures,
                         int* cellCount, 
                         cudaStream_t stream);

void launch_find_index_cell_alive(
    int *world_id,
    int world_dim_tot,
    int *alive_cell_vector,
    int *n_cell_alive_d,
    int *n_cell_alive_h,
    cudaStream_t stream
);
