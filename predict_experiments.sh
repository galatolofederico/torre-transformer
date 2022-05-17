#!/bin/sh

python predict.py predict.model=./models/VectorAutoRegressor/prepost.ckpt predict.split=train dataset=prepost architecture=VectorAutoRegressor
python predict.py predict.model=./models/VectorAutoRegressor/prepost.ckpt predict.split=validation dataset=prepost architecture=VectorAutoRegressor
python predict.py predict.model=./models/VectorAutoRegressor/prepost.ckpt predict.split=test dataset=prepost architecture=VectorAutoRegressor
python predict.py predict.model=./models/VectorAutoRegressor/prepost.ckpt predict.split=post dataset=prepost architecture=VectorAutoRegressor

python predict.py predict.model=./models/VectorAutoRegressor/full.ckpt predict.split=train dataset=full architecture=VectorAutoRegressor
python predict.py predict.model=./models/VectorAutoRegressor/full.ckpt predict.split=validation dataset=full architecture=VectorAutoRegressor
python predict.py predict.model=./models/VectorAutoRegressor/full.ckpt predict.split=test dataset=full architecture=VectorAutoRegressor

python predict.py predict.model=./models/VectorAutoRegressor/post.ckpt predict.split=train dataset=post architecture=VectorAutoRegressor
python predict.py predict.model=./models/VectorAutoRegressor/post.ckpt predict.split=validation dataset=post architecture=VectorAutoRegressor
python predict.py predict.model=./models/VectorAutoRegressor/post.ckpt predict.split=test dataset=post architecture=VectorAutoRegressor


python predict.py predict.model=./models/TransformerRegressor/prepost.ckpt predict.split=train dataset=prepost architecture=TransformerRegressor
python predict.py predict.model=./models/TransformerRegressor/prepost.ckpt predict.split=validation dataset=prepost architecture=TransformerRegressor
python predict.py predict.model=./models/TransformerRegressor/prepost.ckpt predict.split=test dataset=prepost architecture=TransformerRegressor
python predict.py predict.model=./models/TransformerRegressor/prepost.ckpt predict.split=post dataset=prepost architecture=TransformerRegressor

python predict.py predict.model=./models/TransformerRegressor/full.ckpt predict.split=train dataset=full architecture=TransformerRegressor
python predict.py predict.model=./models/TransformerRegressor/full.ckpt predict.split=validation dataset=full architecture=TransformerRegressor
python predict.py predict.model=./models/TransformerRegressor/full.ckpt predict.split=test dataset=full architecture=TransformerRegressor

python predict.py predict.model=./models/TransformerRegressor/post.ckpt predict.split=train dataset=post architecture=TransformerRegressor
python predict.py predict.model=./models/TransformerRegressor/post.ckpt predict.split=validation dataset=post architecture=TransformerRegressor
python predict.py predict.model=./models/TransformerRegressor/post.ckpt predict.split=test dataset=post architecture=TransformerRegressor