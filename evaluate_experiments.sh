#!/bin/sh

tsp -S 2

tsp python evaluate.py evaluate.model=./models/VectorAutoRegressor/prepost.ckpt evaluate.split=train dataset=prepost architecture=VectorAutoRegressor
tsp python evaluate.py evaluate.model=./models/VectorAutoRegressor/prepost.ckpt evaluate.split=validation dataset=prepost architecture=VectorAutoRegressor
tsp python evaluate.py evaluate.model=./models/VectorAutoRegressor/prepost.ckpt evaluate.split=test dataset=prepost architecture=VectorAutoRegressor
tsp python evaluate.py evaluate.model=./models/VectorAutoRegressor/prepost.ckpt evaluate.split=post dataset=prepost architecture=VectorAutoRegressor

tsp python evaluate.py evaluate.model=./models/VectorAutoRegressor/full.ckpt evaluate.split=train dataset=full architecture=VectorAutoRegressor
tsp python evaluate.py evaluate.model=./models/VectorAutoRegressor/full.ckpt evaluate.split=validation dataset=full architecture=VectorAutoRegressor
tsp python evaluate.py evaluate.model=./models/VectorAutoRegressor/full.ckpt evaluate.split=test dataset=full architecture=VectorAutoRegressor

tsp python evaluate.py evaluate.model=./models/VectorAutoRegressor/post.ckpt evaluate.split=train dataset=post architecture=VectorAutoRegressor
tsp python evaluate.py evaluate.model=./models/VectorAutoRegressor/post.ckpt evaluate.split=validation dataset=post architecture=VectorAutoRegressor
tsp python evaluate.py evaluate.model=./models/VectorAutoRegressor/post.ckpt evaluate.split=test dataset=post architecture=VectorAutoRegressor


tsp python evaluate.py evaluate.model=./models/TransformerRegressor/prepost.ckpt evaluate.split=train dataset=prepost architecture=TransformerRegressor
tsp python evaluate.py evaluate.model=./models/TransformerRegressor/prepost.ckpt evaluate.split=validation dataset=prepost architecture=TransformerRegressor
tsp python evaluate.py evaluate.model=./models/TransformerRegressor/prepost.ckpt evaluate.split=test dataset=prepost architecture=TransformerRegressor
tsp python evaluate.py evaluate.model=./models/TransformerRegressor/prepost.ckpt evaluate.split=post dataset=prepost architecture=TransformerRegressor

tsp python evaluate.py evaluate.model=./models/TransformerRegressor/full.ckpt evaluate.split=train dataset=full architecture=TransformerRegressor
tsp python evaluate.py evaluate.model=./models/TransformerRegressor/full.ckpt evaluate.split=validation dataset=full architecture=TransformerRegressor
tsp python evaluate.py evaluate.model=./models/TransformerRegressor/full.ckpt evaluate.split=test dataset=full architecture=TransformerRegressor

tsp python evaluate.py evaluate.model=./models/TransformerRegressor/post.ckpt evaluate.split=train dataset=post architecture=TransformerRegressor
tsp python evaluate.py evaluate.model=./models/TransformerRegressor/post.ckpt evaluate.split=validation dataset=post architecture=TransformerRegressor
tsp python evaluate.py evaluate.model=./models/TransformerRegressor/post.ckpt evaluate.split=test dataset=post architecture=TransformerRegressor