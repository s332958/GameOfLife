#pragma once
const int MaxVisione = 15*15;
struct Cellula {
    int index;
    int center_x;
    int center_y;
    int dim_visione;
    int ID;
    bool alive = 1;
    float visione [MaxVisione];
    float output [10];

    // Costruttore
    __host__ __device__ Cellula(int index = -1, int ID = -1, int dim_mondo, int dim_visione) {
        this->index = index;
        this->ID = ID;
        this->alive = 1;
        this->dim_visione = dim_visione;
        center_x = index % dim_mondo;
        center_y = index / dim_mondo
    }

    __global__ void calcolo_visione();

    void launch_calcolo_visione(float* mondo_cu, float* mondo_signal_cu, int dim_mondo);
};
