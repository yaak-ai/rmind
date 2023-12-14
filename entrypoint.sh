#!/bin/bash

train=false
predict=false
dvc=false
WANDB_MODE=offline
WAND_API_KEY=NA
aws_access_key_id=
aws_secret_access_key=
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
  AWS_ACCESS_KEY_ID=$aws_access_key_id AWS_SECRET_ACCESS_KEY=$aws_secret_access_key dvc repro
else
  echo "¿Neither train, predict or dvc?"
fi
