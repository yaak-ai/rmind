#!/bin/bash

train=false
predict=false
dvc=false
WANDB_MODE=offline
WANDB_API_KEY=NA
flags="--help"

help_message="¿Que? Please specify --train to run model training or --predict to one-off inference"

. parse_args.sh

echo train=$train predict=$predict dvc=$dvc

# Actual entrypoint
if [ $train = true ]; then
  echo WANDB_MODE=$WANDB_MODE WAND_API_KEY=$WAND_API_KEY python train.py $flags
  WANDB_MODE=$WANDB_MODE WAND_API_KEY=$WAND_API_KEY python train.py $flags
elif [ $predict = true ]; then
  echo WANDB_MODE=$WANDB_MODE WAND_API_KEY=$WAND_API_KEY python predict.py $flags
  WANDB_MODE=$WANDB_MODE WAND_API_KEY=$WAND_API_KEY python predict.py $flags
elif [ $dvc = true ]; then
  echo dvc repro
  dvc repro
else
  echo "¿Neither train, predict or dvc?"
fi
