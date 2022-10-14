#!/usr/bin/env bash

cd ..

python main.py -d CollegeMsg --data_usage 1.0 --mode t --n_degree 32 1 --pos_dim 108 --pos_sample multinomial \
--pos_enc saw --temporal_bias 1e-5 --spatial_bias 1.0 --ee_bias 0 --tau 0.05 --negs 9 --solver rk4 --step_size 0.125 \
--bs 32 --gpu 0 --seed 0 --cpu_cores 1