# Real-world benchmarks

![plots](./image.png)

## Overview

We provide the scripts for the download and execution of the real-world benchmarks we used. Most of the scripts in this directory were taken directly from or inspired by "Benchmarking GNNs" by Dwivedi _et al._ refer to their [code](https://github.com/graphdeeplearning/benchmarking-gnns) and [paper](https://arxiv.org/abs/2003.00982) for more details on their work.

- `configs` contains .json configuration files for the various datasets;
- `data` contains scripts to download the datasets and python's classes to load pkl datasets;
- `nets` contains the architectures (layers and nets) that were used with the DGN in the benchmarks;
- `train` contains the training scripts.
  
These benchmarks use the DGL version of DGN (`../models/dgl`)

## Dependencies


## Test run

### Benchmark Setup

[Follow these instructions](./docs/setup.md) to install the benchmark and setup the environment.

### Run model training
```
# at the root of the repo
cd realworld_benchmark
python { main_molecules.py | main_molecules.py } [--param=value ...] --dataset { ZINC | MNIST | CIFAR10 } --gpu_id g͟p͟u͟_͟i͟d͟ --config c͟͟o͟n͟f͟i͟g͟_͟f͟i͟l͟e͟

```


## Tuned hyperparameters

You can find below the hyperparameters we used for our experiments. In general, the depth of the architectures was not changed while the width was adjusted to keep the total number of parameters of the model between 100k and 110k as done in "Benchmarking GNNs" to ensure a fair comparison of the architectures. Refer to our [paper](https://arxiv.org/abs/2004.05718) for an interpretation of the results.

```
--- DGN ---

# ZINC
python main_molecules.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="sum eig1-smooth eig2-smooth" --scalers="identity" --towers=5 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=10
python main_molecules.py --weight_decay=3e-6 --L=4 --hidden_dim=70 --out_dim=60 --residual=True --edge_feat=True --edge_dim=50 --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="sum eig1-smooth eig2-smooth" --scalers="identity" --towers=5 --pretrans_layers=1 --posttrans_layers=1 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=20

python main_molecules.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="sum eig1-smooth eig2-smooth max min" --scalers="identity" --towers=5 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=10
python main_molecules.py --weight_decay=3e-6 --L=4 --hidden_dim=70 --out_dim=60 --residual=True --edge_feat=True --edge_dim=50 --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="sum eig1-smooth eig2-smooth max min" --scalers="identity" --towers=5 --pretrans_layers=1 --posttrans_layers=1 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=20

# CIFAR10
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.1 --graph_norm=True --batch_norm=True --aggregators="sum eig1-smooth eig2-smooth" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_pna_CIFAR10.json" --lr_schedule_patience=5
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=True --edge_dim=50 --readout=sum --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="sum eig1-smooth eig2-smooth" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_pna_CIFAR10.json" --lr_schedule_patience=5

python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.1 --graph_norm=True --batch_norm=True --aggregators="sum eig1-smooth eig2-smooth max min" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_pna_CIFAR10.json" --lr_schedule_patience=5
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=True --edge_dim=50 --readout=sum --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="sum eig1-smooth eig2-smooth max min" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_pna_CIFAR10.json" --lr_schedule_patience=5
