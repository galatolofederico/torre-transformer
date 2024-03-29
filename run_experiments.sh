#!/bin/sh

WANDB=true
TAG=testfull1

mkdir -p ./models

python train.py train.save_model=./models/LSTMRegressor/prepost.ckpt dataset=prepost train.wandb=$WANDB wandb.tag=$TAG architecture=LSTMRegressor
python train.py train.save_model=./models/LSTMRegressor/full.ckpt dataset=full train.wandb=$WANDB wandb.tag=$TAG architecture=LSTMRegressor
python train.py train.save_model=./models/LSTMRegressor/post.ckpt dataset=post train.wandb=$WANDB wandb.tag=$TAG architecture=LSTMRegressor

python train.py train.save_model=./models/VectorAutoRegressor/prepost.ckpt dataset=prepost train.wandb=$WANDB wandb.tag=$TAG architecture=VectorAutoRegressor
python train.py train.save_model=./models/VectorAutoRegressor/full.ckpt dataset=full train.wandb=$WANDB wandb.tag=$TAG architecture=VectorAutoRegressor
python train.py train.save_model=./models/VectorAutoRegressor/post.ckpt dataset=post train.wandb=$WANDB wandb.tag=$TAG architecture=VectorAutoRegressor

python train.py train.save_model=./models/TransformerRegressor/prepost.ckpt dataset=prepost train.wandb=$WANDB wandb.tag=$TAG architecture=TransformerRegressor
python train.py train.save_model=./models/TransformerRegressor/full.ckpt dataset=full train.wandb=$WANDB wandb.tag=$TAG architecture=TransformerRegressor
python train.py train.save_model=./models/TransformerRegressor/post.ckpt dataset=post train.wandb=$WANDB wandb.tag=$TAG architecture=TransformerRegressor