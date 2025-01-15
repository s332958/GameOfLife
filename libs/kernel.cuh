
extern "C" void wrap_add_creature_to_world(float* creature, float *world, int *id_matrix, 
                                        int dim_creature, int dim_world, int pos_x, int pos_y, 
                                        int creature_id, int *number_of_creaure);

extern "C" void wrap_convolution(float *world, int *id_matrix, float* filter, float *world_out, int *id_matrix_out, 
                            int dim_world, int dim_filter, int number_of_creatures);