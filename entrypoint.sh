#!/bin/bash

train=false
predict=false
WANDB_MODE=offline
WAND_API_KEY=NA
flags="--help"

help_message="¿Que? Please specify --train to run model training or --predict to one-off inference"

. parse_args.sh

# Actual entrypoint
if [ $train ]; then
  WANDB_MODE=$WANDB_MODE WAND_API_KEY=$WAND_API_KEY python train.py $flags
elif [ $predict ]; then
  WANDB_MODE=$WANDB_MODE WAND_API_KEY=$WAND_API_KEY python predict.py $flags
else
  echo "¿Neither train nor predict?"
fi
