#!/bin/bash

train=false
predict=false
WANDB_MODE=offline
WANDB_API_KEY=NA
HYDRA_FULL_ERROR=0
flags="--help"

help_message="¿Que? Please specify --train to run model training or --predict to one-off inference"

cd /home/yaak || exit 1

. parse_args.sh

echo train=$train predict=$predict


# Actual entrypoint
if [ $train = true ]; then
  ytt --ignore-unknown-comments -f config/dataset/templates/ --output yaml --output-files config/dataset/
  echo HYDRA_FULL_ERROR=$HYDRA_FULL_ERROR WANDB_MODE=$WANDB_MODE WANDB_API_KEY=$WANDB_API_KEY python train.py $flags
  HYDRA_FULL_ERROR=$HYDRA_FULL_ERROR WANDB_MODE=$WANDB_MODE WANDB_API_KEY=$WANDB_API_KEY python train.py $flags
elif [ $predict = true ]; then
  echo HYDRA_FULL_ERROR=$HYDRA_FULL_ERROR WANDB_MODE=$WANDB_MODE WANDB_API_KEY=$WANDB_API_KEY python predict.py $flags
  HYDRA_FULL_ERROR=$HYDRA_FULL_ERROR WANDB_MODE=$WANDB_MODE WANDB_API_KEY=$WANDB_API_KEY python predict.py $flags
else
  echo "¿Neither train nor predict?"
fi
