#!/bin/bash

# Run this before training to allow write permissiond
# on lightning_logs wandb outputs

echo "Making docker volumes"
echo "mkdir -p lightning_logs wandb outputs"

mkdir -p lightning_logs wandb outputs 
chmod ugo+rwx lightning_logs wandb outputs

ls -al lightning_logs wandb outputs