#include <cuda_runtime.h>

#ifndef KERNEL
#define KERNEL

extern "C" void wrap_add_creature_to_world(float* creature, float *world, int *id_matrix, 
                                        int dim_creature, int dim_world, int pos_x, int pos_y, 
                                        int creature_id, int *number_of_creaure, cudaStream_t stream);

extern "C" void wrap_convolution(Cellula *cellule_cu, float *mondo_creature, float *mondo, int *id_matrix, int dim_world,
                                 int number_of_creatures, int cellCount, int dim_output, int convolution_iter, cudaStream_t stream);

extern "C" void wrap_creature_evaluation(float *world, int *id_matrix, 
                                        int *creature_occupations, float *creature_values, 
                                        int dim_world, int number_of_creatures, cudaStream_t stream);

extern "C" void wrap_add_base_food(float *world, int *id_matrix, float max_food, int dim_world);

extern "C" int wrap_cellule_cleanup(Cellula *cellule_cu, int* cellCount, int* mask_cu);

#endif 