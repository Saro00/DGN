import time
import dgl
import torch
from torch.utils.data import Dataset
import random as rd
from ogb.graphproppred import Evaluator
import networkx as nx
from tqdm import tqdm

import scipy
from scipy import sparse as sp
import numpy as np
import itertools
import torch.utils.data
import pandas as pd
import shutil, os
import os.path as osp
from dgl.data.utils import load_graphs, save_graphs, Subset
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_dgl import read_csv_graph_dgl


def positional_encoding(g, pos_enc_dim, norm):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    if norm == 'none':
        N = sp.diags(g.in_degrees().numpy(), dtype=float)
        L = N * sp.eye(g.number_of_nodes()) - A
    elif norm == 'sym':
        N = sp.diags(g.in_degrees().numpy() ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N
    elif norm == 'walk':
        N = sp.diags(g.in_degrees().numpy() ** -1., dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A

    # Loop all connected components to compute eigenvectors separately
    components = list(nx.connected_components(g.to_networkx().to_undirected()))
    EigVec = torch.zeros((L.shape[0], pos_enc_dim), dtype=torch.float32)
    for component in components:
        comp = list(component)
        this_L = L[comp][:, comp]
        if pos_enc_dim < len(comp) - 1:  # Compute the k-lowest eigenvectors
            this_EigVal, this_EigVec = sp.linalg.eigs(this_L, k=pos_enc_dim, which='SR', tol=1e-5)
        else:  # Compute all eigenvectors
            this_EigVal, this_EigVec = scipy.linalg.eig(this_L.toarray())
            if pos_enc_dim > len(comp):  # Pad with non-sense eigenvectors
                temp_EigVal = np.ones(pos_enc_dim - len(comp), dtype=np.float32) + float('inf')
                temp_EigVec = np.zeros((len(comp), pos_enc_dim - len(comp)), dtype=np.float32)
                this_EigVal = np.concatenate([this_EigVal, temp_EigVal], axis=0)
                this_EigVec = np.concatenate([this_EigVec, temp_EigVec], axis=1)

        # Sort and convert to torch
        this_EigVec = this_EigVec[:, this_EigVal.argsort()]
        this_Eigvec = torch.from_numpy(np.real(this_EigVec[:, :pos_enc_dim])).float()
        EigVec[comp, :] = this_Eigvec

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim, which='SR', tol=1e-5)
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    g.ndata['eig'] = torch.from_numpy(np.real(EigVec[:, :pos_enc_dim])).float()
    # g.ndata['eig'] = torch.from_numpy(np.random.rand(g.number_of_nodes(), pos_enc_dim)).float()
    del A
    del N
    del L
    del EigVec
    del EigVal
    return g


class DownloadPCBA(object):
    """ Modified version of DglGraphPropPredDataset of ogb.graphproppred, that doesn't save the dataset """

    def __init__(self, name='ogbg-pcba', root="data"):
        self.name = name  ## original name, e.g., ogbg-mol-tox21
        self.dir_name = 'ogbg_molpcba_dgl'
        self.original_root = root
        self.root = osp.join(root, self.dir_name)

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.

        self.download_name = 'pcba'  ## name of downloaded file, e.g., tox21

        self.num_tasks = 128
        self.eval_metric = 'ap'
        self.task_type = 'binary classification'
        self.num_classes = 2

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        raw_dir = osp.join(self.root, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        ### download
        url = 'https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/pcba.zip'
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            # delete folder if there exists
            try:
                shutil.rmtree(self.root)
            except:
                pass
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        else:
            print("Stop download.")
            exit(-1)

        ### preprocess
        add_inverse_edge = True
        additional_node_files = []
        additional_edge_files = []

        graphs = read_csv_graph_dgl(raw_dir, add_inverse_edge=add_inverse_edge,
                                    additional_node_files=additional_node_files,
                                    additional_edge_files=additional_edge_files)

        labels = pd.read_csv(osp.join(raw_dir, "graph-label.csv.gz"), compression="gzip", header=None).values

        has_nan = np.isnan(labels).any()

        if "classification" in self.task_type:
            if has_nan:
                labels = torch.from_numpy(labels)
            else:
                labels = torch.from_numpy(labels).to(torch.long)
        else:
            labels = torch.from_numpy(labels)

        print('Not Saving...')
        # save_graphs(pre_processed_file_path, graphs, labels={'labels': labels})

        ### load preprocessed files
        self.graphs = graphs
        self.labels = labels

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = 'scaffold'

        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header=None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header=None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header=None).values.T[0]

        return {"train": torch.tensor(train_idx, dtype=torch.long), "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long)}

    def __getitem__(self, idx):
        """Get datapoint with index"""

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


class PCBADGL(torch.utils.data.Dataset):
    def __init__(self, data, split, norm='none', pos_enc_dim=0):
        self.split = split
        self.data = [g for g in data[self.split]]
        self.graph_lists = []
        self.graph_labels = []
        for i, g in enumerate(self.data):
            if g[0].number_of_nodes() > 5:
                self.graph_lists.append(g[0])
                self.graph_labels.append(g[1])
        self.n_samples = len(self.graph_lists)
        del self.data

    def get_eig(self, norm):

        print('Computing Eigenvectors...')
        with tqdm(range(len(self.graph_lists)), unit='Graph') as t:
            for ii in t:
                self.graph_lists[ii] = positional_encoding(self.graph_lists[ii], 3, norm=norm)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class PCBADataset(Dataset):
    def __init__(self, name, pos_enc_dim=0, norm='none', verbose=True):
        start = time.time()
        if verbose:
            print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.norm = norm
        dataset = DownloadPCBA(name='ogbg-molpcba')
        split_idx = dataset.get_idx_split()
        self.train = PCBADGL(dataset, split_idx['train'], norm=norm, pos_enc_dim=pos_enc_dim)
        self.val = PCBADGL(dataset, split_idx['valid'], norm=norm, pos_enc_dim=pos_enc_dim)
        self.test = PCBADGL(dataset, split_idx['test'], norm=norm, pos_enc_dim=pos_enc_dim)
        del dataset
        del split_idx

        self.evaluator = Evaluator(name='ogbg-molpcba')

        if verbose:
            print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def get_eig_train(self):
        self.train.get_eig(norm=self.norm)

    def get_eig_val(self):
        self.val.get_eig(norm=self.norm)

    def get_eig_test(self):
        self.test.get_eig(norm=self.norm)

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))

        labels = torch.cat([label.unsqueeze(0) for label in labels])
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e