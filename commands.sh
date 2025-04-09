# pretrain carla garage
just train \
    experiment=carla_garage/control_transformer/pretrain.yaml\
    ++wandb.project=devnull\
    trainer.devices="\[0\]"\
    datamodule.train.num_workers=1\
    datamodule.train.batch_size=3

# finetune carla garage
just train \
    experiment=carla_garage/control_transformer/finetune.yaml\
    model.artifact=model-j7owf0g8:v0\
    ++wandb.project=devnull\
    trainer.devices="\[0\]"\
    datamodule.train.num_workers=1\
    datamodule.train.batch_size=3
