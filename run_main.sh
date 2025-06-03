#!/bin/bash

./main.exe \
  -ep 2000 \
  -st 2000 \
  -scale 3 \
  -world_dim 300 \
  -n_creature 30 \
  -max_workspace 10 \
  -eval_method 0 \
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
  -model_structure 162,100,10 \
  -load 

  # -ep INT \
  # -st INT \
  # -scale INT \
  # -world_dim INT \
  # -n_creature INT \
  # -max_workspace INT \
  # -eval_method INT \
  # -reserve_memory INT \
  # -checkpoint_epoch INT \
  # -pn_scale_obstacles FLOAT \
  # -pn_threshold_obstacles FLOAT \
  # -pn_scale_food FLOAT \
  # -pn_threshold_food FLOAT \
  # -random_threshold_food FLOAT \
  # -starting_value FLOAT \
  # -energy_fraction FLOAT \
  # -energy_decay FLOAT \
  # -winners_fraction FLOAT \
  # -recombination_fraction FLOAT \
  # -mutation_probability FLOAT \
  # -mutation_range FLOAT \
  # -clean_window_size INT \
  # -model_structure INT,INT.. \
  # -load \
  # -render \
  # -watch_signaling

