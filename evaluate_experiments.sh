#!/bin/sh

python evaluate.py evaluate.model=./models/prepost.ckpt evaluate.split=train dataset=prepost
python evaluate.py evaluate.model=./models/prepost.ckpt evaluate.split=validation dataset=prepost
python evaluate.py evaluate.model=./models/prepost.ckpt evaluate.split=test dataset=prepost
python evaluate.py evaluate.model=./models/prepost.ckpt evaluate.split=post dataset=prepost

python evaluate.py evaluate.model=./models/full.ckpt evaluate.split=train dataset=full
python evaluate.py evaluate.model=./models/full.ckpt evaluate.split=validation dataset=full
python evaluate.py evaluate.model=./models/full.ckpt evaluate.split=test dataset=full