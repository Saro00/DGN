# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


import torch
import pickle
import torch.utils.data
import time
import numpy as np
import csv
import dgl
from scipy import sparse as sp
import numpy as np

EPS = 1e-5


def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    return g


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        with open(data_dir + "/%s.pickle" % self.split, "rb") as f:
        #with open('data/ZINC.pkl', "rb") as f:

            self.data = pickle.load(f)

        # loading the sampled indices from file ./zinc_molecules/<split>.index
        with open(data_dir + "/%s.index" % self.split, "r") as f:
            data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
            self.data = [self.data[i] for i in data_idx[0]]

        assert len(self.data) == num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"

        """
        data is a list of Molecule dict objects with following attributes
        
          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """

        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()

    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))

        for molecule in self.data:
            node_features = molecule['atom_type'].long()

            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list

            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features


            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features

            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])

    def get_eig(self, pos_enc_dim=7, norm='none'):

        for g in self.graph_lists:
            # Laplacian
            A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
            if norm == 'none':
                N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1), dtype=float)
                L = N * sp.eye(g.number_of_nodes()) - A
            elif norm == 'sym':
                N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
                L = sp.eye(g.number_of_nodes()) - N * A * N
            elif norm == 'walk':
                N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1., dtype=float)
                L = sp.eye(g.number_of_nodes()) - N * A
            EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=5e-1)
            EigVec = EigVec[:, EigVal.argsort()]  # increasing order
            g.ndata['eig'] = torch.from_numpy(np.real(EigVec[:, :pos_enc_dim])).float()

    def _add_positional_encodings(self, pos_enc_dim):

        for g in self.graph_lists:
            g.ndata['pos_enc'] = g.ndata['eig'][:,1:pos_enc_dim+1]


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


class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name

        self.num_atom_type = 28  # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4  # known meta-info about the zinc dataset; can be calculated as well

        data_dir = './data/molecules'

        self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
        self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
        self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time() - t0))


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name, norm='none', pos_enc_dim=0, verbose=True):
        """
            Loading SBM datasets
        """
        start = time.time()
        if verbose:
            print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/'
        with open(data_dir + name + '.pkl', "rb") as f:
            f = pickle.load(f)
            f[0].get_eig(6, norm)
            if pos_enc_dim > 0:
                f[0]._add_positional_encodings(pos_enc_dim)
            self.train = f[0]
            f[1].get_eig(6, norm)
            if pos_enc_dim > 0:
                f[1]._add_positional_encodings(pos_enc_dim)
            self.val = f[1]
            f[2].get_eig(6, norm)
            if pos_enc_dim > 0:
                f[2]._add_positional_encodings(pos_enc_dim)
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        if verbose:
            print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]






def get_laplacian_matrix(adj, norm):
    r"""
    Get the Laplacian/normalized Laplacian matrices from a batch of adjacency matrices
    Parameters
    --------------
        adj: tensor(..., N, N)
            Batches of symmetric adjacency matrices

        norm: string
            Whether to normalize the Laplacian matrix
            If `none`, then `L = D - A`
            If `walk`, then `L = D^-1 (D - A)`
            If 'sym' then 'L = D^-1/2 (D - A) D^-1/2
    Returns
    -------------
        L: tensor(..., N, N)
            Resulting Laplacian matrix
    """

    # Apply the equation L = D - A
    N = adj.shape[-1]
    arr = torch.arange(N)
    L = -adj
    D = torch.sum(adj, dim=-1)
    L[..., arr, arr] = D

    # Normalize by the degree : L = D^-1 (D - A)
    if norm == 'none':
        pass
    elif norm == 'walk':
        Dinv = torch.zeros_like(L)
        Dinv[..., arr, arr] = D ** -1
        L = torch.matmul(Dinv, L)
    elif norm == 'sym':
        Dinv = torch.zeros_like(L)
        Dinv[..., arr, arr] = D ** (-1/2)
        L = torch.matmul(torch.matmul(Dinv, L), Dinv)
    else:
        raise Exception

    return L

def get_k_lowest_eig(adj, k, norm='none'):
    r"""
    Compute the k-lowest eigenvectors of the Laplacian matrix
    for each connected components of the graph. If there are disconnected
    graphs, then the first k eigenvectors are computed for each sub-graph
    separately.
    Parameters
    --------------
        adj: tensor(..., N, N)
            Batches of symmetric adjacency matrices
        k: int
            Compute the k-th smallest eigenvectors and eigenvalues.

        normalize_L: bool
            Whether to normalize the Laplacian matrix
            If `False`, then `L = D - A`
            If `True`, then `L = D^-1 (D - A)`
    Returns
    -------------
        eigvec: tensor(..., N, k)
            Resulting k-lowest eigenvectors of the Laplacian matrix of each sub-graph,
            with the same batching as the `adj` tensor.
            The dim==-1 represents the k-th vectors.
            The dim==-2 represents the N elements of each vector.
            If the a given graph is disconnected, it will give the first ``k`` eigenvector
            of each sub-graph, and will force the first eigenvector to be 0-vectors.
            If there are ``m`` eigenvectors for a given sub-graph, with ``m < k``, it will
            return 0-vectors for all eigenvectors ``> m``
    """

    # Reshape as a 3D tensor for easier looping along batches
    device = adj.device
    shape = list(adj.shape)
    if adj.ndim == 2:
        adj = adj.unsqueeze(0)
    elif adj.ndim > 3:
        adj = adj.view(-1, shape[-2], shape[-1])

    L = get_laplacian_matrix(adj, norm)

    # Compute and sort the eigenvectors

    eigval_all, eigvec_all = torch.symeig(L, eigenvectors=True)
    eigval_all = eigval_all.to(device)
    eigvec_all = eigvec_all.to(device)
    sort_idx = torch.argsort(eigval_all.abs(), dim=-1, descending=False)
    sort_idx_vec = sort_idx.unsqueeze(-2).expand(eigvec_all.shape)
    eigval_sort = torch.gather(eigval_all, dim=-1, index=sort_idx)
    eigvec_sort = torch.gather(eigvec_all, dim=-1, index=sort_idx_vec)

    k_lowest_eigvec = []

    # Loop each graph to detect if some of them are disconnected. If they are disconnected,
    # then modify the eigenvectors such that the lowest k eigenvectors are returned for
    # each sub-graph
    for ii in range(adj.shape[0]):
        this_eigval = eigval_sort[ii]
        num_connected = torch.sum(this_eigval.abs() < EPS)

        # If there is a single connected graph, then return the k lowest eigen functions
        if num_connected <= 1:
            this_eigvec = eigvec_sort[ii, :, :k]
            if k > this_eigvec.shape[-1]:
                temp_eigvec = torch.zeros(this_eigvec.shape[0], k)
                temp_eigvec[:, :k] = this_eigvec
                this_eigvec = temp_eigvec
            k_lowest_eigvec.append(this_eigvec)


        # Otherwise, return the k lowest eigen functions for each sub-graph
        elif num_connected > 1:
            eigvec0 = eigvec_sort[ii, :, :num_connected]
            unique_idx = torch.zeros(1)
            factor = 100

            # Use the eigenvectors with 0 eigenvalues to find the unique sub-graphs
            # And loop to make sure the number of detected sub-graphs is consistent with the
            # Number of connected sub-graphs.
            while (max(unique_idx) + 1) != num_connected:
                eigvec0_round = torch.round(eigvec0 / (factor * EPS))
                _, unique_idx = torch.unique(eigvec0_round, return_inverse=True, dim=0)

                if (max(unique_idx) + 1) < num_connected:
                    factor = (factor / 2)
                elif (max(unique_idx) + 1) > num_connected:
                    factor = (factor * 3)

            # Find the eigenvectors associated to each sub-graph
            sub_graph_factors = torch.zeros(num_connected, len(this_eigval))
            for sub_ii in range(num_connected):
                sub_idx = torch.where(unique_idx == sub_ii)[0]
                sub_graph_factors[sub_ii, :] = torch.mean(torch.abs(eigvec_sort[ii, sub_idx, :]), dim=-2)
            max_idx = torch.argmax(sub_graph_factors, dim=0)[num_connected:]

            # Concatenate the k lowest eigenvectors of each sub-graph
            this_k_lowest_eigvec = torch.zeros(len(this_eigval), k)
            for sub_ii in range(num_connected):
                sub_idx = torch.where(unique_idx == sub_ii)[0]
                k_lowest_idx = torch.where(max_idx == sub_ii)[0][:k - 1] + num_connected
                for kk_enum, kk in enumerate(k_lowest_idx):
                    this_k_lowest_eigvec[sub_idx, kk_enum + 1] = eigvec_sort[ii, sub_idx, kk]

            k_lowest_eigvec.append(this_k_lowest_eigvec)

    # Stack and Reshape to match the input batch shape
    k_lowest_eigvec = torch.stack(k_lowest_eigvec, dim=0).view(*(shape[:-2] + [-1, k]))

    return k_lowest_eigvec.to('cuda' if torch.cuda.is_available() else 'cpu')