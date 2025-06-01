
#include "simulazione.h"
#include "libs/mappa_colori.cuh"
#include "libs/utils_cpu.h"

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
const int n_layer = 4;
int model_structure [n_layer] = {18, 10, 10, 10};
float * weights_models = nullptr; 
float * biases_models = nullptr; 

bool load = false;

size_t reserve_free_memory = 1024 * 1024 * 300; // * 1024;// 1GB

const int MAX_CREATURE = 64;

GLFWwindow* window;
GLuint textureID;

int main(int argc, char* argv[]) {

    // dipendenza temporale ai valori randomici in CPU
    srand(time(0));

    clock_t start = clock();  // Start time 
    bool render = false;
    int numero_epoch = 4;
    int numero_step = 100;
    int scale = 1;
    int world_dim = 200;
    int n_creature = 30;
    int MAX_WORKSPACE = 10000;
    int METHOD_EVAL = 1;
    
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
        if(arg == "-world_dim"){
            world_dim = std::atoi(argv[i+1]);
        }
        if(arg == "-n_creature"){
            n_creature = std::atoi(argv[i+1]);
            if(n_creature>MAX_CREATURE) n_creature = MAX_CREATURE;
        }
        if(arg == "-max_workspace"){
            MAX_WORKSPACE = std::atoi(argv[i+1]);
        }
        if(arg == "-eval_method"){
            METHOD_EVAL = std::atoi(argv[i+1]);
            if(METHOD_EVAL!= 0 || METHOD_EVAL != 1) METHOD_EVAL = 0;
        }
        if(arg == "-reserve_memory"){
            // reserve Mb of free memory for don't sature the GPU
            reserve_free_memory = std::atoi(argv[i+1]) * 1024 * 1024;
        }
        if(arg == "-load"){
            // reserve Mb of free memory for don't sature the GPU
            load = std::atoi(argv[i+1]) != 0;
        }
    }
    
    //fuori dal ciclo
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

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);  // Usa tutta la finestra


        // Proiezione ortografica
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Texture config
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, world_dim, world_dim, 0, GL_RGB, GL_FLOAT, nullptr);

        //creazione colori
        load_constant_memory_GPU(n_creature);
    }

    if (load){
        int n_weight = 0;
        int n_bias = 0;
    
        for(int i = 0; i < n_layer - 1; ++i) {
            n_weight += model_structure[i] * model_structure[i + 1];
            n_bias += model_structure[i + 1];
        }
        size_t tot_models_weight_size = n_creature * n_weight * sizeof(float);
        size_t tot_models_bias_size = n_creature * n_bias * sizeof(float);
    
        weights_models = (float*) malloc(tot_models_weight_size);
        biases_models = (float*) malloc(tot_models_bias_size);
    
        load_model_from_file("models/file1.txt", weights_models, biases_models, n_weight, n_bias, n_creature);
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