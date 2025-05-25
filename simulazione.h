// simulazione.h
#pragma once

#include <cstddef>     // size_t
#include <GLFW/glfw3.h> // GLFWwindow
#include <GL/gl.h>      // GLuint

void simulazione(
    int world_dim, int n_creature, 
    int* model_structure, int n_layer, size_t reserve_free_memory, 
    float* weights_models, float* biases_models, 
    int N_EPHOCS, int N_STEPS, int MAX_WORKSPACE, int METHOD_EVAL, bool render,
    GLFWwindow* window, GLuint textureID
);
