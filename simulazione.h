// simulazione.h
#pragma once

#include <cstddef>     // size_t
#include <GLFW/glfw3.h> // GLFWwindow
#include <GL/gl.h>      // GLuint
#include <vector>

#include <vector>
#include <cstddef> // size_t

struct Simulation_setup {
    int world_dim;
    int n_creature;
    std::vector<int> model_structure;
    int n_layer;
    size_t reserve_free_memory;
    float* weights_models;
    float* biases_models;
    int N_EPHOCS;
    int N_STEPS;
    int MAX_WORKSPACE;
    int METHOD_EVAL;
    bool render;
    bool load;
    int scale;
    int checkpoint_epoch;
    float PN_scale_obstacles;
    float PN_threshold_obstacles;
    float PN_scale_food;
    float PN_threshold_food;
    float random_threshold_food;
    float starting_value;
    float energy_fraction;
    float energy_decay;
    float winners_fraction;
    float recombination_newborns_fraction;
    float gen_x_block;
    float mutation_probability;
    float mutation_range;
    int clean_window_size;

    Simulation_setup(
        int world_dim = 400,
        int n_creature = 10,
        std::vector<int> model_structure = {81, 50, 30, 20, 10},
        int n_layer = 5,
        size_t reserve_free_memory = 30 * 1024 * 1024,
        float* weights_models = nullptr,
        float* biases_models = nullptr,
        int N_EPHOCS = 10,
        int N_STEPS = 100,
        int METHOD_EVAL = 0,
        bool render = false,
        int checkpoint_epoch = 1,
        float PN_scale_obstacles = 10.0f,
        float PN_threshold_obstacles = 0.85f,
        float PN_scale_food = 8.0f,
        float PN_threshold_food = 0.90f,
        float random_threshold_food = 0.99f,
        float starting_value = 9.0f,
        float energy_fraction = (1.0f / 9.0f) / 2.0f,
        float energy_decay = 0.001f,
        float winners_fraction = 0.45f,
        float recombination_newborns_fraction = 0.75f,
        float gen_x_block = 0.005f,
        float mutation_probability = 0.02f,
        float mutation_range = 0.4f,
        int clean_window_size = 11,
        bool load = false,
        int scale = 1
    )
        : world_dim(world_dim),
          n_creature(n_creature),
          model_structure(model_structure),
          n_layer(n_layer),
          reserve_free_memory(reserve_free_memory),
          weights_models(weights_models),
          biases_models(biases_models),
          N_EPHOCS(N_EPHOCS),
          N_STEPS(N_STEPS),
          MAX_WORKSPACE(world_dim * world_dim),
          METHOD_EVAL(METHOD_EVAL),
          render(render),
          checkpoint_epoch(checkpoint_epoch),
          PN_scale_obstacles(PN_scale_obstacles),
          PN_threshold_obstacles(PN_threshold_obstacles),
          PN_scale_food(PN_scale_food),
          PN_threshold_food(PN_threshold_food),
          random_threshold_food(random_threshold_food),
          starting_value(starting_value),
          energy_fraction(energy_fraction),
          energy_decay(energy_decay),
          winners_fraction(winners_fraction),
          recombination_newborns_fraction(recombination_newborns_fraction),
          gen_x_block(gen_x_block),
          mutation_probability(mutation_probability),
          mutation_range(mutation_range),
          clean_window_size(clean_window_size),
          load(load),
          scale(scale)
    {}
};

void simulazione(
    Simulation_setup simulation_setup,
    GLFWwindow* window, GLuint textureID
);
