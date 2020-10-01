# Real-world benchmarks

![plots](./image.png)

## Overview

We provide the scripts for the download and execution of the real-world benchmarks we used. Most of the scripts in this directory were taken directly from or inspired by "Benchmarking GNNs" by Dwivedi _et al._ refer to their [code](https://github.com/graphdeeplearning/benchmarking-gnns) and [paper](https://arxiv.org/abs/2003.00982) for more details on their work.

- `configs` contains .json configuration files for the various datasets;
- `data` contains scripts to download the datasets and python's classes to load pkl datasets;
- `nets` contains the architectures (layers and nets) that were used with the DGN in the benchmarks;
- `train` contains the training scripts.
  
All of the benchmarks use the DGL version of DGN (`../models/dgl`)

## Dependencies


## Test run

### Benchmark Setup

[Follow these instructions](./docs/setup.md) to install the benchmark and setup the environment.

### Run model training
```
# at the root of the repo
cd realworld_benchmark
python { main_molecules.py | main_SBMs_node_classification.py | main_molecules.py } [--param=value ...] --dataset { ZINC | PATTERN | CIFAR10 } --gpu_id g͟p͟u͟_͟i͟d͟ --config c͟͟o͟n͟f͟i͟g͟_͟f͟i͟l͟e͟

```

### Fair comparison

You can find below the scripts used to run the fair comparison between the DGN models. 

```
--- DGN ---

# ZINC
# simple
python main_molecules.py --type="simple" --lap_norm="none" --weight_decay=1e-8 --dropout=0.0 --L=4 --hidden_dim=80 --out_dim=80 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="simple" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=63 --out_dim=63 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="simple" --lap_norm="none" --weight_decay=1e-8 --dropout=0.0 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx dgn2-dx" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="simple" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=63 --out_dim=63 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="simple" --lap_norm="none" --weight_decay=1e-8 --dropout=0.0 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn2-smooth" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="simple" --lap_norm="none" --weight_decay=1e-8 --dropout=0.0 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn2-smooth" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="simple" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn1-dx" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
# complex
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=65 --out_dim=65 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=45 --out_dim=45 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx dgn2-dx" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=45 --out_dim=45 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn2-smooth" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=45 --out_dim=45 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx dgn1-smooth" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
# complex with edge features
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=65 --out_dim=65 --residual=True --edge_feat=True --edge_dim=10  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=True --edge_dim=10  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=45 --out_dim=45 --residual=True --edge_feat=True --edge_dim=10  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx dgn2-dx" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=True --edge_dim=10  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=45 --out_dim=45 --residual=True --edge_feat=True --edge_dim=10  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn2-smooth" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20
python main_molecules.py --type="complex" --lap_norm="none" --weight_decay=3e-6 --dropout=0.0 --L=4 --hidden_dim=45 --out_dim=45 --residual=True --edge_feat=True --edge_dim=10  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx dgn1-smooth" --scalers="identity attenuation amplification" --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20

# PATTERN
# simple
python main_SBMs_node_classification.py --type='simple' --lap_norm='none' --weight_decay=1e-8 --L=4 --hidden_dim=80 --out_dim=80 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean" --scalers="identity attenuation amplification" --dataset='SBM_PATTERN' --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
python main_SBMs_node_classification.py --type="simple" --lap_norm="none" --weight_decay=1e-8 --L=4 --hidden_dim=63 --out_dim=63 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx" --scalers="identity attenuation amplification" --dataset="SBM_PATTERN" --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
python main_SBMs_node_classification.py --type='simple' --lap_norm='none' --weight_decay=1e-8 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx dgn2-dx" --scalers="identity attenuation amplification" --dataset='SBM_PATTERN' --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
python main_SBMs_node_classification.py --type="simple" --lap_norm="none" --weight_decay=1e-8 --L=4 --hidden_dim=63 --out_dim=63 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth" --scalers="identity attenuation amplification" --dataset="SBM_PATTERN" --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
python main_SBMs_node_classification.py --type='simple' --lap_norm='none' --weight_decay=1e-8 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn2-smooth " --scalers="identity attenuation amplification" --dataset='SBM_PATTERN' --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
python main_SBMs_node_classification.py --type="simple" --lap_norm="none" --weight_decay=1e-8 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn1-dx" --scalers="identity attenuation amplification" --dataset="SBM_PATTERN" --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
# complex
python main_SBMs_node_classification.py --type='complex' --lap_norm='none' --weight_decay=1e-8 --L=4 --hidden_dim=55 --out_dim=55 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean" --scalers="identity attenuation amplification" --dataset='SBM_PATTERN' --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
python main_SBMs_node_classification.py --type="complex" --lap_norm="none" --weight_decay=1e-8 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx" --scalers="identity attenuation amplification" --dataset="SBM_PATTERN" --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
python main_SBMs_node_classification.py --type="complex" --lap_norm="none" --weight_decay=1e-8 --L=4 --hidden_dim=47 --out_dim=47 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx dgn2-dx" --scalers="identity attenuation amplification" --dataset="SBM_PATTERN" --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
python main_SBMs_node_classification.py --type="complex" --lap_norm="none" --weight_decay=1e-8 --L=4 --hidden_dim=52 --out_dim=52 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth" --scalers="identity attenuation amplification" --dataset="SBM_PATTERN" --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
python main_SBMs_node_classification.py --type="complex" --lap_norm="none" --weight_decay=1e-8 --L=4 --hidden_dim=47 --out_dim=47 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn2-smooth" --scalers="identity attenuation amplification" --dataset="SBM_PATTERN" --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
python main_SBMs_node_classification.py --type="complex" --lap_norm="none" --weight_decay=1e-8 --L=4 --hidden_dim=47 --out_dim=47 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn1-dx" --scalers="identity attenuation amplification" --dataset="SBM_PATTERN" --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5

# CIFAR10
# simple
python main_superpixels.py --type="simple" --lap_norm="none" --coord_eig --weight_decay=1e-8 --L=4 --hidden_dim=145 --out_dim=145 --residual=True --edge_feat=False --readout=mean --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean" --scalers="identity" --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_DGN_CIFAR10.json" --lr_schedule_patience=5
python main_superpixels.py --type="simple" --lap_norm="none" --coord_eig --weight_decay=1e-8 --L=4 --hidden_dim=90 --out_dim=90 --residual=True --edge_feat=False --readout=mean --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx dgn2-dx" --scalers="identity" --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_DGN_CIFAR10.json" --lr_schedule_patience=5
python main_superpixels.py --type="simple" --lap_norm="none" --coord_eig --weight_decay=1e-8 --L=4 --hidden_dim=90 --out_dim=90 --residual=True --edge_feat=False --readout=mean --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn2-smooth" --scalers="identity" --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_DGN_CIFAR10.json" --lr_schedule_patience=5
# complex
python main_superpixels.py --type="complex" --lap_norm="none" --coord_eig --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=75 --residual=True --edge_feat=False --readout=mean --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean" --scalers="identity" --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_DGN_CIFAR10.json" --lr_schedule_patience=5
python main_superpixels.py --type="complex" --lap_norm="none" --coord_eig --weight_decay=3e-6 --L=4 --hidden_dim=65 --out_dim=65 --residual=True --edge_feat=False --readout=mean --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx dgn2-dx" --scalers="identity" --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_DGN_CIFAR10.json" --lr_schedule_patience=5
python main_superpixels.py --type="complex" --lap_norm="none" --coord_eig --weight_decay=3e-6 --L=4 --hidden_dim=65 --out_dim=65 --residual=True --edge_feat=False --readout=mean --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn2-smooth" --scalers="identity" --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_DGN_CIFAR10.json" --lr_schedule_patience=5

```


## Tuned hyperparameters

You can find below the scripts of the best fine-tuned DGN model for each dataset.

```
# ZINC

python main_molecules.py --type="towers" --lap_norm="none" --weight_decay=3e-7 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=True --edge_dim=10 --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx max min" --scalers="identity amplification attenuation" --towers=5 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_DGN_ZINC.json" --lr_schedule_patience=20

# CIFAR10

python main_superpixels.py --type="towers" --coord_eig --weight_decay=3e-6 --L=4 --hidden_dim=100 --out_dim=95 --residual=True --edge_feat=True --edge_dim=10 --readout=sum --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean dgn1-dx dgn2-dx max" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_DGN_CIFAR10.json" --lr_schedule_patience=5

# PATTERN

python main_SBMs_node_classification.py --type="complex" --lap_norm="none" --weight_decay=1e-8 --L=4 --hidden_dim=47 --out_dim=47 --residual=True --edge_feat=False  --readout=mean --graph_norm=True --batch_norm=True --aggregators="mean dgn1-smooth dgn1-dx" --scalers="identity attenuation amplification" --dataset="SBM_PATTERN" --gpu_id 0 --config "configs/SBMs_node_clustering_DGN_PATTERN.json" --lr_schedule_patience=5
```
