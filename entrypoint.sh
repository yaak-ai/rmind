#!/bin/bash -xe

# Print debug information for GPU environments
if command -v nvidia-smi &> /dev/null
then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi

# Actual entrypoint
if [ "$1" = '--train' ]; then
  exec python train.py "${@:2}"
elif [ "$1" = '--predict' ]; then
  exec python predict.py "${@:2}"
else
  echo "Please specify --train to run model training or "\
    "--predict to one-off inference"
fi

exec "$@"
