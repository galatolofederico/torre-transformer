#!/bin/sh

WANDB=true

mkdir -p ./models

python train.py train.save_model=./models/prepost.ckpt dataset=prepost train.wandb=$WANDB
python train.py train.save_model=./models/full.ckpt dataset=full train.wandb=$WANDB