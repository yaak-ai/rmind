#!/bin/bash

# Run this after the training to return the ownership from uid=1000 to the current user


sudo chown -R $(id -un):$(id -gn) lightning_logs wandb outputs 
ls -al lightning_logs wandb outputs