#!/bin/bash

echo START!

eval "$(conda shell.bash hook)"
conda activate python_env

for pfn_number in {0..10}; do
    torchrun --nnodes=1 --nproc-per-node=1 pfns4mvbo/evaluation.py -t regression -p BODi -m "$pfn_number" -s 1234
    torchrun --nnodes=1 --nproc-per-node=1 pfns4mvbo/evaluation.py -t overlap -p BODi -m "$pfn_number" -s 1234
    torchrun --nnodes=1 --nproc-per-node=1 pfns4mvbo/evaluation.py -t optimization -p BODi -m "$pfn_number" -s 1234
done 

echo DONE!
