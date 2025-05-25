// Perlin noise – versione semplificata per CUDA
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

__device__ int perm[256] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
    140,36,103,30,69,142,8,99,37,240,21,10,23,151,160,137,
    91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,
    30,69,142,8,99,37,240,21,10,23, /* ...ripeti fino a 256 elementi */
};

__device__ float perlin(float x, float y) {
    int X = (int)floorf(x) & 255;
    int Y = (int)floorf(y) & 255;

    x -= floorf(x);
    y -= floorf(y);

    float u = fade(x);
    float v = fade(y);

    int A = perm[X] + Y;
    int B = perm[X + 1] + Y;

    return lerp(
        lerp(grad(perm[A], x, y), grad(perm[B], x - 1, y), u),
        lerp(grad(perm[A + 1], x, y - 1), grad(perm[B + 1], x - 1, y - 1), u),
        v
    );
}

// Kernel: applica il Perlin noise + soglia
__global__ void perlinNoiseThreshold_kernel(
    int* world_id_d, int world_dim,
    float scale, float threshold,
    float* offset_x, float* offset_y)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = world_dim * world_dim;
    if (idx >= total) return;

    int x = idx % world_dim;
    int y = idx / world_dim;

    float nx = (x + offset_x) * scale;
    float ny = (y + offset_y) * scale;

    float noise = perlin(nx, ny);
    noise = (noise + 1.0f) * 0.5f; // normalizza tra 0 e 1

    world_id_d[idx] = (noise > threshold) ? -1 : 0;


}

int* d_world;
int world_dim = 128;

void perlinNoiseThreshold(int world_dim, 
    int* world_id_d, int num_creature){
        float scale = 0.05f;
        float threshold = 0.55f;
        int total = world_dim * world_dim;
        float offset_x = rand() % 10000;
        float offset_y = rand() % 10000;
            
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        perlinNoiseThreshold_kernel<<<blocks, threads>>>(world_id_d, world_dim, scale, threshold, offset_x, offset_y);
        cudaDeviceSynchronize();

    }
