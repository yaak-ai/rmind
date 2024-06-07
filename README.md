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

## Embeddings
```
python predict.py inference=embeddings paths.metadata_cache_dir=./yaak-datasets/metadata ++datamodule.predict.num_workers=2 model.base.artifact=yaak/cargpt/model-ojp6qagn:v10 ++trainer.devices=[1] ++output_dir=/home/harsimrat/workspace/embeddings/cargpt/model-ojp6qagn:v10/train
```

## Faiss

```
find /home/harsimrat/workspace/embeddings/cargpt/model-ojp6qagn:v10/train -type f -name '*.pt' > /home/harsimrat/workspace/embeddings/cargpt/model-ojp6qagn:v10/train.txt
shuf /home/harsimrat/workspace/embeddings/cargpt/model-ojp6qagn:v10/train.txt > /home/harsimrat/workspace/embeddings/cargpt/model-ojp6qagn:v10/train-shuffle.txt
```

```
python scripts/build-index.py -l /home/harsimrat/workspace/embeddings/cargpt/model-ojp6qagn:v10/train-shuffle.txt -x 10000 -s 0.2 -i /home/harsimrat/workspace/faiss/cargpt/model-ojp6qagn:v10/faiss-action-summary.index -m /home/harsimrat/workspace/faiss/cargpt/model-ojp6qagn:v10/faiss-action-summary.metadata -o forward_dynamics -t observation_summary
```

For searching in the index checkout the [notebook]('notebook/faiss.ipynb')
