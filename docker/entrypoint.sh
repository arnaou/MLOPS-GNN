#!/bin/sh
echo 'login to wandb'
wandb login $SECRET_KEY
echo 'initate dvc pull'
dvc pull
echo 'dvc pull complete'
#python -u src/data/make_dataset.py
echo 'make dataset complete'
python -u src/models/train_model.py
echo 'training complete'
python -u src/models/predict_model.py
echo 'predict complete'
