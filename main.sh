#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --config $1

python utils/sortout_res.py $1