#include "libs/kernel.cu"
#include "libs/utils.cuh"

int main() {
    int dim_world = 64;
    int raggio = 32;
    int input_workspace = raggio * raggio;

    // Host allocation
    float *h_world_value = new float[dim_world * dim_world];
    int   *h_world_id = new int[dim_world * dim_world];
    float *h_world_signaling = new float[dim_world * dim_world];
    int   *h_cell_idx = new int[1] {0};
    float *h_input = new float[input_workspace];

    // Initialize dummy data
    for (int i = 0; i < dim_world * dim_world; ++i) {
        h_world_value[i] = 1.0f;
        h_world_signaling[i] = 2.0f;
        h_world_id[i] = i;
    }

    // Device allocation
    float *d_world_value, *d_world_signaling, *d_input;
    int *d_world_id, *d_cell_idx;
    cudaMalloc(&d_world_value, dim_world * dim_world * sizeof(float));
    cudaMalloc(&d_world_signaling, dim_world * dim_world * sizeof(float));
    cudaMalloc(&d_world_id, dim_world * dim_world * sizeof(int));
    cudaMalloc(&d_cell_idx, sizeof(int));
    cudaMalloc(&d_input, input_workspace * sizeof(float));

    // Copy to device
    cudaMemcpy(d_world_value, h_world_value, dim_world * dim_world * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_world_signaling, h_world_signaling, dim_world * dim_world * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_world_id, h_world_id, dim_world * dim_world * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cell_idx, h_cell_idx, sizeof(int), cudaMemcpyHostToDevice);

    // Launch wrapper
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launch_vision(d_world_value, d_world_id, d_world_signaling, dim_world, d_cell_idx, raggio, input_workspace, d_input, stream);
    cudaStreamSynchronize(stream);

    // Copy result back
    cudaMemcpy(h_input, d_input, input_workspace * sizeof(float), cudaMemcpyDeviceToHost);

    // Print part of the result
    std::cout << "Output (first 10 values): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    delete[] h_world_value;
    delete[] h_world_signaling;
    delete[] h_world_id;
    delete[] h_cell_idx;
    delete[] h_input;
    cudaFree(d_world_value);
    cudaFree(d_world_signaling);
    cudaFree(d_world_id);
    cudaFree(d_cell_idx);
    cudaFree(d_input);
    cudaStreamDestroy(stream);

    return 0;
}