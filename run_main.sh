#!/bin/bash
nsys profile
compute-sanitizer --tool memcheck --log-file sanitazer_report.txt  ./main.exe \
  -ep 100 \
  -st 40 \
  -scale 5 \
  -world_dim 256 \
  -n_creature 2 \
  -eval_method 0 \
  -reserve_memory 50 \
  -checkpoint_epoch 50 \
  -pn_scale_obstacles 14.5 \
  -pn_threshold_obstacles 0.8 \
  -pn_scale_food 10.0 \
  -pn_threshold_food 0.80 \
  -random_threshold_food 0.90 \
  -starting_value 128.0 \
  -energy_fraction 1.0 \
  -energy_decay 0.0005 \
  -winners_fraction 0.15 \
  -recombination_fraction 0.95 \
  -mutation_probability 1.0 \
  -mutation_range 0.04 \
  -clean_window_size 3 \
  -model_structure 50,100,10 \
  -load \
  -render \

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

