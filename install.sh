#!/bin/bash

conda create -n temporal python=3.8
conda activate temporal
pip install -r requirements.txt

mkdir data
cd data
wget https://storage.googleapis.com/gresearch/templama/train.json
wget https://storage.googleapis.com/gresearch/templama/val.json
wget https://storage.googleapis.com/gresearch/templama/test.json
cd ../
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz
pip install en_core_web_sm-3.7.0.tar.gz