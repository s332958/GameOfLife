#!/bin/bash
nsys profile
compute-sanitizer --tool memcheck --log-file sanitazer_report.txt  ./main.exe \
  -ep 100 \
  -st 2000 \
  -scale 8 \
  -world_dim 128 \
  -n_creature 15 \
  -eval_method 0 \
  -reserve_memory 50 \
  -checkpoint_epoch 10 \
  -pn_scale_obstacles 15.5 \
  -pn_threshold_obstacles 0.8 \
  -pn_scale_food 12.0 \
  -pn_threshold_food 0.85 \
  -random_threshold_food 0.90 \
  -starting_value 20.0 \
  -energy_fraction 0.08 \
  -energy_decay 0.0005 \
  -winners_fraction 0.3 \
  -recombination_fraction 0.90 \
  -mutation_probability 1.0 \
  -mutation_range 0.05 \
  -clean_window_size 5 \
  -model_structure 162,243,162,81,10 \
  -render \
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

