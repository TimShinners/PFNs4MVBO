#!/bin/bash

echo START!

eval "$(conda shell.bash hook)"
conda activate python_env

for seed in {5000..5029}; do
  # for opt_id in pfn_bodi_24__ei__is pfn_casmopolitan_16__ei__is__tr pfn_cocabo_51__ei__mab pfn_mixed_5__ei__is; do
  # for opt_id in CoCaBO Casmopolitan BODi random; do
    for task in schwefel michalewicz griewank rosenbrock levy xgboost_opt; do
      torchrun --nnodes=1 --nproc-per-node=4 pfns4mvbo/run_experiments.py -t BO -oi $opt_id -lt $task -nbos 1 -s $seed
    done
  done
done
echo DONE!