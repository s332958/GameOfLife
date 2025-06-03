#!/bin/bash

./main.exe \
  -ep 50 \
  -st 200 \
  -scale 3 \
  -world_dim 512 \
  -n_creature 20 \
  -max_workspace 10 \
  -eval_method 1 \
  -reserve_memory 50 \
  -checkpoint_epoch 5 \
  -pn_scale_obstacles 15.5 \
  -pn_threshold_obstacles 0.8 \
  -pn_scale_food 12.0 \
  -pn_threshold_food 0.88 \
  -random_threshold_food 0.95 \
  -starting_value 10.0 \
  -energy_fraction 0.1 \
  -energy_decay 0.002 \
  -winners_fraction 0.5 \
  -recombination_fraction 0.7 \
  -mutation_probability 0.03 \
  -mutation_range 0.5 \
  -clean_window_size 15 \
  -model_structure 100,50,25,10

    # -render \
