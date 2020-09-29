# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


import os
import pickle
from scipy.spatial.distance import cdist
from scipy import sparse as sp
import numpy as np
import itertools

import dgl
import torch
import torch.utils.data

import time

import csv
from sklearn.model_selection import StratifiedShuffleSplit
EPS = 1e-5



def sigma(dists, kth=8):
    # Compute sigma and reshape
    try:
        # Get k-nearest neighbors for each node
        knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1))/kth
    except ValueError:     # handling for graphs with num_nodes less than kth
        num_nodes = dists.shape[0]
        # this sigma value is irrelevant since not used for final compute_edge_list
        sigma = np.array([1]*num_nodes).reshape(num_nodes,1)
        
    return sigma + 1e-8 # adding epsilon to avoid zero value of sigma


def compute_adjacency_matrix_images(coord, feat, use_feat=True, kth=8):
    coord = coord.reshape(-1, 2)
    # Compute coordinate distance
    c_dist = cdist(coord, coord)
    
    if use_feat:
        # Compute feature distance
        f_dist = cdist(feat, feat)
        # Compute adjacency
        A = np.exp(- (c_dist/sigma(c_dist))**2 - (f_dist/sigma(f_dist))**2 )
    else:
        A = np.exp(- (c_dist/sigma(c_dist))**2)
        
    # Convert to symmetric matrix
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    return A        


def compute_edges_list(A, kth=8+1):
    # Get k-similar neighbor indices for each node

    num_nodes = A.shape[0]
    new_kth = num_nodes - kth
    
    if num_nodes > 9:
        knns = np.argpartition(A, new_kth-1, axis=-1)[:, new_kth:-1]
        knn_values = np.partition(A, new_kth-1, axis=-1)[:, new_kth:-1] # NEW
    else:
        # handling for graphs with less than kth nodes
        # in such cases, the resulting graph will be fully connected
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_values = A # NEW
        
        # removing self loop
        if num_nodes != 1:
            knn_values = A[knns != np.arange(num_nodes)[:,None]].reshape(num_nodes,-1) # NEW
            knns = knns[knns != np.arange(num_nodes)[:,None]].reshape(num_nodes,-1)
    return knns, knn_values # NEW


class SuperPixDGL(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 dataset,
                 split,
                 use_mean_px=True,
                 use_coord=True,
                 proportion=1.):

        self.split = split
        
        self.graph_lists = []
        
        if dataset == 'MNIST':
            self.img_size = 28
            with open(os.path.join(data_dir, 'mnist_75sp_%s.pkl' % split), 'rb') as f:
                self.labels, self.sp_data = pickle.load(f)
                self.graph_labels = torch.LongTensor(self.labels)
        elif dataset == 'CIFAR10':
            self.img_size = 32
            with open(os.path.join(data_dir, 'cifar10_150sp_%s.pkl' % split), 'rb') as f:
                self.labels, self.sp_data = pickle.load(f)
                self.graph_labels = torch.LongTensor(self.labels)
                print()
                
        self.use_mean_px = use_mean_px
        self.use_coord = use_coord
        self.n_samples = len(self.labels)
        self.proportion = proportion
        
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))
        self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = [], [], [], []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord = sample[:2]
            
            try:
                coord = coord / self.img_size
            except AttributeError:
                VOC_has_variable_image_sizes = True
                
            if self.use_mean_px:
                A = compute_adjacency_matrix_images(coord, mean_px) # using super-pixel locations + features
            else:
                A = compute_adjacency_matrix_images(coord, mean_px, False) # using only super-pixel locations
            edges_list, edge_values_list = compute_edges_list(A) # NEW

            N_nodes = A.shape[0]
            
            mean_px = mean_px.reshape(N_nodes, -1)
            coord = coord.reshape(N_nodes, 2)
            x = np.concatenate((mean_px, coord), axis=1)

            edge_values_list = edge_values_list.reshape(-1) # NEW # TO DOUBLE-CHECK !
            
            self.node_features.append(x)
            self.edge_features.append(edge_values_list) # NEW
            self.Adj_matrices.append(A)
            self.edges_lists.append(edges_list)
        
        for index in range(len(self.sp_data)):
            g = dgl.DGLGraph()
            g.add_nodes(self.node_features[index].shape[0])
            g.ndata['feat'] = torch.Tensor(self.node_features[index]).half()


            for src, dsts in enumerate(self.edges_lists[index]):
                # handling for 1 node where the self loop would be the only edge
                # since, VOC Superpixels has few samples (5 samples) with only 1 node
                if self.node_features[index].shape[0] == 1:
                    g.add_edges(src, dsts)
                else:
                    g.add_edges(src, dsts[dsts!=src])
            
            # adding edge features for Residual Gated ConvNet
            edge_feat_dim = g.ndata['feat'].shape[1] # dim same as node feature dim
            #g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim).half() 
            g.edata['feat'] = torch.Tensor(self.edge_features[index]).unsqueeze(1).half()  # NEW 

            self.graph_lists.append(g)

    def get_eig(self, coord_eig):
        if coord_eig:
            self.graph_lists = [coord_encoding(g) for g in self.graph_lists]
        else:
            self.graph_lists = [positional_encoding(g, 7) for g in self.graph_lists]
            self.graph_lists = sort_list_eig(self.graph_lists)


        #for g in self.graph_lists:
            #A = g.adjacency_matrix().to_dense()
            #g.ndata['eig'] = get_k_lowest_eig(A, 7)

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


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def get_eig(self, coord_eig):
        if coord_eig:
            self.graph_lists = [coord_encoding(g) for g in self.graph_lists]
        else:
            self.graph_lists = [positional_encoding(g, 7) for g in self.graph_lists]
            self.graph_lists = sort_list_eig(self.graph_lists)
        #for g in self.graph_lists:
            #A = g.adjacency_matrix().to_dense()
            #g.ndata['eig'] = get_k_lowest_eig(A, 7)


    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])
    
    
class SuperPixDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name, num_val=5000):
        """
            Takes input standard image dataset name (MNIST/CIFAR10) 
            and returns the superpixels graph.
            
            This class uses results from the above SuperPix class.
            which contains the steps for the generation of the Superpixels
            graph from a superpixel .pkl file that has been given by
            https://github.com/bknyaz/graph_attention_pool
            
            Please refer the SuperPix class for details.
        """
        t_data = time.time()
        self.name = name

        use_mean_px = True # using super-pixel locations + features
        use_mean_px = False # using only super-pixel locations
        if use_mean_px:
            print('Adj matrix defined from super-pixel locations + features')
        else:
            print('Adj matrix defined from super-pixel locations (only)')
        use_coord = True
        self.test = SuperPixDGL("./data/superpixels", dataset=self.name, split='test',
                            use_mean_px=use_mean_px, 
                            use_coord=use_coord)

        self.train_ = SuperPixDGL("./data/superpixels", dataset=self.name, split='train',
                             use_mean_px=use_mean_px, 
                             use_coord=use_coord)

        _val_graphs, _val_labels = self.train_[:num_val]
        _train_graphs, _train_labels = self.train_[num_val:]

        self.val = DGLFormDataset(_val_graphs, _val_labels)
        self.train = DGLFormDataset(_train_graphs, _train_labels)

        print("[I] Data load time: {:.4f}s".format(time.time()-t_data))
        


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in SuperPixDataset class.
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

    

class SuperPixDataset(torch.utils.data.Dataset):

    def __init__(self, name, coord_eig=False, proportion=1., verbose=True):
        """
            Loading Superpixels datasets
        """
        start = time.time()
        if verbose:
            print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            print("Total graphs training set ", len(f[0]))

            if proportion < 1. - 1e-5:
                l = int(len(f[0])*proportion)
                #f[0].lists = f[0].lists[:l]
                #f[0].graph_lists = f[0].graph_lists[:l]
                #f[0].graph_labels = f[0].graph_labels[:l]
                f[0] = DGLFormDataset(f[0].graph_lists[:l], f[0].graph_labels[:l])

            print("Number of graphs used for training ", len(f[0]))

            f[0].get_eig(coord_eig)
            self.train = f[0]
            f[1].get_eig(coord_eig)
            self.val = f[1]
            f[2].get_eig(coord_eig)
            self.test = f[2]
        if verbose:
            print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]
        
        self.train = DGLFormDataset(self.train.graph_lists, self.train.graph_labels)
        self.val = DGLFormDataset(self.val.graph_lists, self.val.graph_labels)
        self.test = DGLFormDataset(self.test.graph_lists, self.test.graph_labels)

                            



def get_laplacian_matrix(adj, normalize_L):
    r"""
    Get the Laplacian/normalized Laplacian matrices from a batch of adjacency matrices
    Parameters
    --------------
        adj: tensor(..., N, N)
            Batches of symmetric adjacency matrices

        normalize_L: bool
            Whether to normalize the Laplacian matrix
            If `False`, then `L = D - A`
            If `True`, then `L = D^-1 (D - A)`
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
    if normalize_L:
        Dinv = torch.zeros_like(L)
        Dinv[..., arr, arr] = D ** -1
        L = torch.matmul(Dinv, L)

    return L

def get_k_lowest_eig(adj, k):
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

    L = get_laplacian_matrix(adj, normalize_L=False)

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

def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR') # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    g.ndata['eig'] = torch.from_numpy(np.real(EigVec[:,:pos_enc_dim])).float()

    return g

def get_scores(x, y, eig):
    n = x.shape[0]
    hor = 0
    ver = 0
    for i in range(n):
        if float(eig[i]) > 0:
            if float(x[i]) > 0.5:
                hor += 1
            else:
                hor -= 1
            if float(y[i]) > 0.5:
                ver += 1
            else:
                ver -= 1

    scores = {}
    scores['hor'] = abs(hor)
    scores['ver'] = abs(ver)

    return scores

def sort_eig(graph):
    x = graph.ndata['feat'][:, 3]
    y = graph.ndata['feat'][:, 4]
    eigs = graph.ndata['eig']
    eig1 = eigs[:, 1]
    eig2 = eigs[:, 2]
    scores1 = get_scores(x, y, eig1)
    scores2 = get_scores(x, y, eig2)

    if scores1['hor'] == max(scores1['hor'], scores2['ver'], scores1['ver'], scores2['hor']):
        return graph
    elif scores2['ver'] == max(scores1['hor'], scores2['ver'], scores1['ver'], scores2['hor']):
        return graph
    elif scores1['ver'] == max(scores1['hor'], scores2['ver'], scores1['ver'], scores2['hor']):
        eigs[:, 1] = eig2
        eigs[:, 2] = eig1
        graph.ndata['eig'] = eigs
        return graph
    else:
        eigs[:, 1] = eig2
        eigs[:, 2] = eig1
        graph.ndata['eig'] = eigs
        return graph

def sort_list_eig(list):
    list_new = [sort_eig(graph) for graph in list]
    return list_new

def coord_encoding(graph):
    x = graph.ndata['feat'][:, 3:4].type(torch.FloatTensor)
    y = graph.ndata['feat'][:, 4:5].type(torch.FloatTensor)
    null = torch.zeros(x.shape).type(torch.FloatTensor)
    graph.ndata['eig'] = torch.cat([null, x, y], dim=-1)
    return graph