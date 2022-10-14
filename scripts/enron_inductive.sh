#!/usr/bin/env bash

cd ..

python main.py -d enron --data_usage 0.5 --mode i --n_degree 64 1 --pos_dim 108 --pos_sample multinomial --pos_enc saw \
--temporal_bias 1e-5 --spatial_bias 0.001 --ee_bias 0.01 --tau 0.05 --negs 2 --solver rk4 --step_size 0.125 \
--bs 32 --gpu 0 --seed 0 --cpu_cores 1 --limit_ngh_span --ngh_span 320 8
