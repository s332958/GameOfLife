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
                         float* contribution_matrix, 
                         int* cells,
                         int world_dim, 
                         int number_of_creatures,
                         int* cellCount, 
                         cudaStream_t stream);

// Wrapper per ripulire celle disattivate
void launch_cellule_cleanup(int* cells, int* cellCount, int* id_matrix, cudaStream_t stream);
