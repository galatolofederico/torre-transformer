#!/bin/sh

WANDB=true

mkdir -p ./models

python train.py train.save_model=./models/VectorAutoRegressor/prepost.ckpt dataset=prepost train.wandb=$WANDB architecture=VectorAutoRegressor
python train.py train.save_model=./models/VectorAutoRegressor/full.ckpt dataset=full train.wandb=$WANDB architecture=VectorAutoRegressor

python train.py train.save_model=./models/TransformerRegressor/prepost.ckpt dataset=prepost train.wandb=$WANDB architecture=TransformerRegressor
python train.py train.save_model=./models/TransformerRegressor/full.ckpt dataset=full train.wandb=$WANDB architecture=TransformerRegressor