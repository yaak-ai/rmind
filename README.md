# carGPT

Self-supervised model trained on vehicle context and control signals from expert drives.

## Setup
```
just install
```

## Training

### SMART
```bash
just train experiment=smart [++trainer.fast_dev_run=1 ...]
```

## Inference

1. start rerun and open [http://localhost:9090/?url=ws://localhost:9877](http://localhost:9090/?url=ws://localhost:9877)
```bash
just rerun
```

3. in another terminal:
```bash
just predict inference=smart model.artifact=<wandb_path>
```
