# Directional Graph Network

Implementation of Directional Graph Network for Graph Nets in PyTorch and DGL.

## Overview

We provide the implementation of the Directional Graph Network (DGN) in PyTorch and DGL frameworks, along with scripts for running real-world benchmarks. The repository is organised as follows:

- `models` contains:
  - `pytorch` contains the various GNN models implemented in PyTorch:
    - the implementation of the aggregators, the scalers, the DGN layer and the computation of the directional aggregation matrix ('eigen_agg').
  - `dgl` contains the DGN model implemented via the [DGL library](https://www.dgl.ai/): aggregators, scalers, and DGN layer.
  - `layers.py` contains general NN layers used by the various models
- `realworld_benchmark` contains various scripts from [Benchmarking GNNs](https://github.com/graphdeeplearning/benchmarking-gnns) to download the real-world benchmarks and train the DGN on them. In `real_world/README.md` we provide instructions for runnning the experiments.


## Reference
```
@article{corso2020principal,
  title={Principal Neighbourhood Aggregation for Graph Nets},
  author={Corso, Gabriele and Cavalleri, Luca and Beaini, Dominique and Lio, Pietro and Velickovic, Petar},
  journal={arXiv preprint arXiv:2004.05718},
  year={2020}
}
```

## License
MIT
