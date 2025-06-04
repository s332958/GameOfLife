#ifndef PERLIN_NOISE_CUH
#define PERLIN_NOISE_CUH

// Function for generating food and obstacles in perling form
void launch_perlinNoise_obstacles(int world_dim, int* world_id_d, float scale, float threshold, cudaStream_t stream);
void launch_perlinNoise_food(int world_dim, int* world_id_d, float* world_d, float scale, float threshold, cudaStream_t stream);

#endif 
