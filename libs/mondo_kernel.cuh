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
                         float* world_signal,
                         float* contribution_matrix, 
                         int world_dim, 
                         int number_of_creatures,
                         float energy_decay,
                         cudaStream_t stream);

// function for compute the new array cell alive, that found index of alive cell and then compact them in a vector without space                         
void launch_find_index_cell_alive(
    int *world_id,
    int world_dim_tot,
    int *alive_cell_vector,
    int *n_cell_alive_d,
    int *n_cell_alive_h,
    int *support_vector_d,
    cudaStream_t stream
);

// function for clean che surrounding of the creature (this is for balance the start of each creature)
void launch_clean_around_cells(
    float* world_value_d, 
    int* world_id_d, 
    int dim_world, 
    int* cellule, 
    int* ncellule, 
    int window_size,
    cudaStream_t stream);

void compact_with_thrust(int* mondo_id, int* alive_cells_d, int world_dim, int &new_size);
