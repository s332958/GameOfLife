#ifndef PERLIN_NOISE_CUH
#define PERLIN_NOISE_CUH

// Funzione host che genera ostacoli usando Perlin noise
void launch_perlinNoise_obstacles(int world_dim, int* world_id_d, cudaStream_t stream);
void launch_perlinNoise_food(int world_dim, int* world_id_d, float* world_d, cudaStream_t stream);

#endif // PERLIN_NOISE_CUH
