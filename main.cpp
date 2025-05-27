
#include "simulazione.h"
#include "libs/mappa_colori.cuh"

#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <format>

//cudaMallocAsync e cudaFreeAsync disponibili solo su GPU con Compute Capability >= 7.0
const int world_dim = 300;
const int n_creature = 5;
const int n_layer = 5;
int model_structure [n_layer] = {18, 1, 1, 1, 10};
float * weights_models = nullptr; 
float * biases_models = nullptr; 
int const METHOD_EVAL = 0;

size_t reserve_free_memory = 1024 * 1024 * 300; // * 1024;// 1GB
int const MAX_WORKSPACE = 10000;

const int MAX_CREATURE = 64;

GLFWwindow* window;
GLuint textureID;

int main(int argc, char* argv[]) {
    clock_t start = clock();  // Start time 
    bool render = false;
    int numero_epoch = 1;
    int numero_step = 1;
    int scale = 1;
    
    // Controlla se c'è almeno un argomento
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-render") {
            render = true;
        }
        if(arg == "-ep"){
            numero_epoch = std::atoi(argv[i+1]);
        }
        if(arg == "-st"){
            numero_step = std::atoi(argv[i+1]);
        }
        if(arg == "-scale"){
            scale = std::atoi(argv[i+1]);
        }
    }
    
    //creazione finestra opengl
    if(render){
        if (!glfwInit()) {
            std::cerr << "Errore nell'inizializzazione di GLFW" << std::endl;
            return -1;
        }
        window = glfwCreateWindow(world_dim*scale, world_dim*scale, "OpenGL Image Rendering", NULL, NULL);
    
        if (!window) {
            std::cerr << "Errore nella creazione della finestra" << std::endl;
            glfwTerminate();
            return -1;
        }
    
        glfwMakeContextCurrent(window);

        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        //creazione colori
        load_constant_memory_GPU();
    }


    simulazione(
        world_dim, n_creature, 
        model_structure, n_layer, reserve_free_memory, 
        weights_models, biases_models, 
        numero_epoch, numero_step, MAX_WORKSPACE, METHOD_EVAL, render,
        window, textureID
    );

    if(render){
        glDeleteTextures(1, &textureID);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    



    clock_t end = clock();  // End time
    std::cout << "Tempo esecuzione programma: " << (end - start) / CLOCKS_PER_SEC << " secondi" << std::endl;

    return 0;
}