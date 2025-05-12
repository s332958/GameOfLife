#pragma once

struct Cellula {
    int index;
    int ID;
    bool alive = 1;
    float visione [Size*Size];
    float output [10];

    // Costruttore
    __host__ __device__ Cellula(int index = -1, int ID = -1)
        : index(index), ID(ID), alive(1) {}

    __global__ void kernel_visione();
};
