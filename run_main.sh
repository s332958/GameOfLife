#!/bin/bash
nvprof ./main.exe \
  -ep 5 \
  -st 1000 \
  -scale 1 \
  -world_dim 1024 \
  -n_creature 100 \
  -eval_method 1 \
  -reserve_memory 50 \
  -checkpoint_epoch 1 \
  -pn_scale_obstacles 14.5 \
  -pn_threshold_obstacles 0.8 \
  -pn_scale_food 10.0 \
  -pn_threshold_food 0.90 \
  -random_threshold_food 0.99 \
  -starting_value 128.0 \
  -energy_fraction 1.0 \
  -energy_decay 0.006 \
  -clone_fraction 0.7 \
  -clean_window_size 3 \
  -model_structure 50,25,10\
  -mutation_range 6 \
  -learning_rate 1 \
  -max_workspace 10 \
  -render \
  -load 

  # -render \
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
  # -load BOOL\
  # -render BOOL\
  # -watch_signaling BOOL \
  # -alpha FLOAT \
  # -std FLOAT


# nsys profile ./main.exe \
# compute-sanitizer --tool memcheck --log-file sanitazer_report.txt  ./main.exe \