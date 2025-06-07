
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

GLFWwindow* window;
GLuint textureID;

void parse_args(int argc, char** argv, Simulation_setup& setup) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-render") {
            setup.render = true;
        } else if (arg == "-ep" && i + 1 < argc) {
            setup.N_EPOCH = std::stoi(argv[++i]);
        } else if (arg == "-st" && i + 1 < argc) {
            setup.N_STEPS = std::stoi(argv[++i]);
        } else if (arg == "-scale" && i + 1 < argc) {
            setup.scale = std::stoi(argv[++i]);
        } else if (arg == "-world_dim" && i + 1 < argc) {
            setup.world_dim = std::stoi(argv[++i]);
            setup.MAX_WORKSPACE = setup.world_dim * setup.world_dim;
        } else if (arg == "-n_creature" && i + 1 < argc) {
            setup.n_creature = std::stoi(argv[++i]);
        } else if (arg == "-max_workspace" && i + 1 < argc) {
            setup.MAX_WORKSPACE = std::stoi(argv[++i]);
        } else if (arg == "-eval_method" && i + 1 < argc) {
            int method = std::stoi(argv[++i]);
            if (method == 0 || method == 1) {
                setup.METHOD_EVAL = method;
            }else{
                setup.METHOD_EVAL = 0;
            }
        } else if (arg == "-reserve_memory" && i + 1 < argc) {
            setup.reserve_free_memory = static_cast<size_t>(std::stoi(argv[++i])) * 1024 * 1024;
        } else if (arg == "-load") {
            setup.load = true;
        } else if (arg == "-checkpoint_epoch" && i + 1 < argc) {
            setup.checkpoint_epoch = std::stoi(argv[++i]);
        } else if (arg == "-pn_scale_obstacles" && i + 1 < argc) {
            setup.PN_scale_obstacles = std::stof(argv[++i]);
        } else if (arg == "-pn_threshold_obstacles" && i + 1 < argc) {
            setup.PN_threshold_obstacles = std::stof(argv[++i]);
        } else if (arg == "-pn_scale_food" && i + 1 < argc) {
            setup.PN_scale_food = std::stof(argv[++i]);
        } else if (arg == "-pn_threshold_food" && i + 1 < argc) {
            setup.PN_threshold_food = std::stof(argv[++i]);
        } else if (arg == "-random_threshold_food" && i + 1 < argc) {
            setup.random_threshold_food = std::stof(argv[++i]);
        } else if (arg == "-starting_value" && i + 1 < argc) {
            setup.starting_value = std::stof(argv[++i]);
        } else if (arg == "-energy_fraction" && i + 1 < argc) {
            setup.energy_fraction = std::stof(argv[++i]);
        } else if (arg == "-energy_decay" && i + 1 < argc) {
            setup.energy_decay = std::stof(argv[++i]);
        } else if (arg == "-winners_fraction" && i + 1 < argc) {
            setup.winners_fraction = std::stof(argv[++i]);
        } else if (arg == "-recombination_fraction" && i + 1 < argc) {
            setup.recombination_newborns_fraction = std::stof(argv[++i]);
        } else if (arg == "-mutation_probability" && i + 1 < argc) {
            setup.mutation_probability = std::stof(argv[++i]);
        } else if (arg == "-mutation_range" && i + 1 < argc) {
            setup.mutation_range = std::stof(argv[++i]);
        } else if (arg == "-clean_window_size" && i + 1 < argc) {
            setup.clean_window_size = std::stoi(argv[++i]);
        } else if (arg == "-watch_signaling") {
            setup.watch_signaling = true;
        } else if (arg == "-model_structure" && i + 1 < argc) {
            std::string list = argv[++i];
            setup.model_structure.clear();
            std::stringstream ss(list);
            std::string num;
            while (std::getline(ss, num, ',')) {
                try {
                    setup.model_structure.push_back(std::stoi(num));
                } catch (...) {
                    std::cerr << "Invalid model_structure value: " << num << std::endl;
                }
            }
            setup.n_layer = static_cast<int>(setup.model_structure.size());
            int offset = 0;
            for(int i=0; i<setup.n_layer; i++){
                if(i<setup.n_layer-1) offset += sprintf(setup.file_model + offset, "%d_", setup.model_structure[i]);
                else offset += sprintf(setup.file_model + offset, "%d.txt", setup.model_structure[i]);
            }
            printf("Model file: %s \n",setup.file_model);
        }else if ( arg == "-alpha" && i + 1 < argc){
            setup.alpha = std::stof(argv[++i]);
        }else if (arg == "-std" && i + 1 < argc){
            setup.std = std::stof(argv[++i]);
        }
    }
}

int main(int argc, char* argv[]) {

    clock_t start = clock();  // Start time 
    // dipendenza temporale ai valori randomici in CPU
    srand(time(0));

    Simulation_setup simulation_setup = Simulation_setup();
    
    parse_args(argc, argv, simulation_setup);
    
    //fuori dal ciclo
    if(simulation_setup.render){
        if (!glfwInit()) {
            std::cerr << "Error initialization of GLFW" << std::endl;
            return -1;
        }
        window = glfwCreateWindow(simulation_setup.world_dim*simulation_setup.scale, simulation_setup.world_dim*simulation_setup.scale, "OpenGL Image Rendering", NULL, NULL);
    
        if (!window) {
            std::cerr << "Error in creating window" << std::endl;
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, simulation_setup.world_dim, simulation_setup.world_dim, 0, GL_RGB, GL_FLOAT, nullptr);

        //creazione colori
        load_constant_memory_GPU(simulation_setup.n_creature);
    }

    // if load is present, start loading creature from model/structure_of_the_model
    if (simulation_setup.load){
        int n_weight = 0;
        int n_bias = 0;
    
        for(int i = 0; i < simulation_setup.n_layer - 1; ++i) {
            n_weight += simulation_setup.model_structure[i] * simulation_setup.model_structure[i+1];
            n_bias += simulation_setup.model_structure[i + 1];
        }
        size_t tot_model_weight_size = n_weight * sizeof(float);
        size_t tot_model_bias_size =  n_bias * sizeof(float);
    
        simulation_setup.weights_models = (float*) malloc(tot_model_weight_size);
        simulation_setup.biases_models = (float*) malloc(tot_model_bias_size);
        
        char path_file[300];
        sprintf(path_file,"models/%s",simulation_setup.file_model);
        bool load_done = load_model_from_file(path_file, simulation_setup.weights_models, simulation_setup.biases_models, n_weight, n_bias);
        
        // if load of creature is not possible free the memory and set nullprt to weights and biases
        if(load_done==false){
            free(simulation_setup.weights_models);
            free(simulation_setup.biases_models);
            simulation_setup.weights_models = nullptr;
            simulation_setup.biases_models = nullptr;
        }
    
    }else{        

        // generation of score_file
        std::ofstream out("log_score.txt", std::ios::trunc); 
        out.close();

    }

    // reduce max_workspace to world_dimension
    if(simulation_setup.world_dim*simulation_setup.world_dim<simulation_setup.MAX_WORKSPACE){
        simulation_setup.MAX_WORKSPACE = simulation_setup.world_dim*simulation_setup.world_dim;
    }

    simulazione(
        simulation_setup,
        window, textureID
    );

    if(simulation_setup.render){
        glDeleteTextures(1, &textureID);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    clock_t end = clock();  // End time
    std::cout << "Tempo esecuzione programma: " << (end - start) / CLOCKS_PER_SEC << " secondi" << std::endl;

    return 0;
}