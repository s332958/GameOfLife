#include "perlin_noise.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <ctime>
#include <iostream>

// permutation vector
__device__ int perm[256] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
    140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
    247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
    57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
    60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
    65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
    200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
    52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
    207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
    119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
    218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
    81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
    184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
    222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

__device__ float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float grad(int hash, float x, float y) {
    int h = hash & 7;
    float u = h < 4 ? x : y;
    float v = h < 4 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
}

__device__ float perlin(float x, float y) {
    int X = (int)floorf(x) % 255;
    int Y = (int)floorf(y) % 255;

    x -= floorf(x);
    y -= floorf(y);

    float u = fade(x);
    float v = fade(y);

    int A = (perm[X] + Y) % 255;
    int B = (perm[X + 1] + Y) % 255;

    float x1 = lerp(grad(perm[A], x, y), grad(perm[B], x - 1, y), u);
    float x2 = lerp(grad(perm[A + 1], x, y - 1), grad(perm[B + 1], x - 1, y - 1), u);

    return lerp(x1, x2, v);
}

__global__ void perlinNoise_obstacles_kernel(int world_dim, int* world_id_d, float scale, float threshold, float x_offset, float y_offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= world_dim || y >= world_dim) return;

    int index = y * world_dim + x;

    float nx = (x + x_offset) / scale;
    float ny = (y + y_offset) / scale;

    float val = perlin(nx, ny);
    val = (val + 1.0f) * 0.5f;
    if  (world_id_d[index] > 0 || val < threshold) return;

    world_id_d[index] = -1;
}

__global__ void perlinNoise_food_kernel(int world_dim, int* world_id_d, float* world_d, float scale, float threshold, float x_offset, float y_offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= world_dim || y >= world_dim) return;

    int index = y * world_dim + x;

    float nx = (x + x_offset) / scale;
    float ny = (y + y_offset) / scale;

    float val = perlin(nx, ny);
    val = (val + 1.0f) * 0.5f;
    if  (world_id_d[index] != 0 || val < threshold) return;

    world_d[index] = 20;
}

void launch_perlinNoise_obstacles(int world_dim, int* world_id_d, float scale, float threshold, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((world_dim + 15) / 16, (world_dim + 15) / 16);

    float x_offset = (float)(clock() % 10000);
    float y_offset = (float)((clock() * 31) % 10000);

    perlinNoise_obstacles_kernel<<<gridSize, blockSize, 0, stream>>>(world_dim, world_id_d, scale, threshold, x_offset, y_offset);
}

void launch_perlinNoise_food(int world_dim, int* world_id_d, float* world_d, float scale, float threshold, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((world_dim + 15) / 16, (world_dim + 15) / 16);
    
    float x_offset = (float)(clock() % 10000);
    float y_offset = (float)((clock() * 31) % 10000);

    perlinNoise_food_kernel<<<gridSize, blockSize, 0, stream>>>(world_dim, world_id_d, world_d, scale, threshold, x_offset, y_offset);
}
