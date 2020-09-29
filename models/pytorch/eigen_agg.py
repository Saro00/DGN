import torch
import scipy as sp
import networkx as nx


EPS = 1e-5

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

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

    eigval_all, eigvec_all = torch.symeig(L.cpu(), eigenvectors=True)
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
                k_lowest_idx = torch.where(max_idx == sub_ii)[0][:k-1] + num_connected
                for kk_enum, kk in enumerate(k_lowest_idx):
                    this_k_lowest_eigvec[sub_idx, kk_enum+1] = eigvec_sort[ii, sub_idx, kk]
            
            k_lowest_eigvec.append(this_k_lowest_eigvec)

    # Stack and Reshape to match the input batch shape
    k_lowest_eigvec = torch.stack(k_lowest_eigvec, dim=0).view(*(shape[:-2] + [-1, k]))

    return k_lowest_eigvec

    


def get_k_lowest_eig_old(adj, k, normalize_L):
    r"""
    Compute the k-lowest eigenvalues and eigenvectors of the Laplacian matrix
    for each connected components of the graph.
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
        eigval: tensor(..., k)
            Resulting k-lowest eigenvalues of the Laplacian matrix,
            with the same batching as the `adj` tensor
        eigvec: tensor(..., N, k)
            Resulting k-lowest eigenvectors of the Laplacian matrix,
            with the same batching as the `adj` tensor.
            The dim==-1 represents the k-th vectors.
            The dim==-2 represents the N elements of each vector. 
    """

    device = adj.device
    # Reshape as a 3D tensor for easier looping along batches
    shape = list(adj.shape)
    if adj.ndim == 2:
        adj = adj.unsqueeze(0)
    elif adj.ndim > 3:
        adj = adj.view(-1, shape[-2], shape[-1])

    L = get_laplacian_matrix(adj, normalize_L=False)
    
    eigvec = torch.zeros(adj.shape[0],adj.shape[-1],k).to(device)
    for ii in range(adj.shape[0]):
        #get singular values decomposition
        u,s,v = torch.svd(L[ii])
        #get 0 singular values positions in svd
        null_mask = (s <= EPS)
        #get basis of 0 singular space
        zero_eig_vect = torch.transpose(v[null_mask],0,1)
        assert zero_eig_vect.shape[0] == adj.shape[-1]
        
        
        components = []
        for jj in range(zero_eig_vect.shape[-1]):
            #get the indices of the connected components by getting nonzero entries in the 
            #0 singular vector associated to the component
            compo = torch.nonzero(zero_eig_vect[:,jj]).squeeze()
            components.append(compo)

        #get the eigenvalues and eigenvectors of each components
        ii_eigvec = torch.zeros(adj.shape[-1],k).to(device)
        for jj, compo in enumerate(components):
            #select the sub matrix corresponding to the component
            sub_adj = adj[ii,:,:][compo].T[compo].T
            
            #if a component has only one node, all eigen vectors are 0
            if sub_adj.shape==torch.Size([]):
                compo_eigvec = torch.zeros(1,k)
                ii_eigvec[compo,:] = compo_eigvec
            else:
                dim = sub_adj.shape[0]
                #if the components Laplacian has less eigenvectors than k, add zero vectors in padding
                if k > dim:
                    compo_eigval, compo_eigvec = get_k_lowest_eig(sub_adj, dim, normalize_L)
                    compo_eigvec = torch.cat((compo_eigvec, torch.zeros(1, dim, k-dim, device=device)), dim=2)

                else:
                    compo_eigval, compo_eigvec = get_k_lowest_eig_connected(sub_adj, k, normalize_L)
                
                #place the k eigenvectors of each component in a vector using the indices of each nodes 
                #of the components nodes
                ii_eigvec[compo,:] = compo_eigvec
        eigvec[ii,:,:]=ii_eigvec

    #makes sense for connected graphs but not for disconnected ones so it is left empty. Remove later
    eigval = []

    return eigval, eigvec

def get_k_lowest_eig_connected_old(adj, k, normalize_L):
    r"""
    Compute the k-lowest eigenvalues and eigenvectors of the Laplacian matrix of a connected graph.
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
        eigval: tensor(..., k)
            Resulting k-lowest eigenvalues of the Laplacian matrix,
            with the same batching as the `adj` tensor
        eigvec: tensor(..., N, k)
            Resulting k-lowest eigenvectors of the Laplacian matrix,
            with the same batching as the `adj` tensor.
            The dim==-1 represents the k-th vectors.
            The dim==-2 represents the N elements of each vector. 
    """

    # Reshape as a 3D tensor for easier looping along batches
    device = adj.device
    shape = list(adj.shape)
    if adj.ndim == 2:
        adj = adj.unsqueeze(0)
    elif adj.ndim > 3:
        adj = adj.view(-1, shape[-2], shape[-1])
    
    L = get_laplacian_matrix(adj, normalize_L=normalize_L)
    eigvec = []
    eigval = []

    # Compute all the k-th smallest eigenvectors and eigenvalues
    for ii in range(L.shape[0]):
        Laplacian = sp.sparse.csr_matrix(L[ii, :, :].clone().detach().cpu().numpy())
        w, v = sp.sparse.linalg.eigsh(Laplacian, k=k, which='SM', tol=0)
        eigval.append(torch.Tensor(w).to(device))
        eigvec.append(torch.Tensor(v).to(device))

    # Stack and Reshape to match the input batch shape
    eigval = torch.stack(eigval, dim=0).view(*(shape[:-2] + [k]))
    eigvec = torch.stack(eigvec, dim=0).view(*(shape[:-2] + [-1, k]))

    return eigval, eigvec


def get_adjacency_from_gradient_of_features(adj, features, normalization='none', add_diag=True, absolute_adj=False):
    r"""
    Computes an adjacency matrix from the gradient of the ``features`` on the graph described
    by the adjacency matrix ``adj``, with different options of normalization. 
    The gradient is a function on the edges, computed from these node features.
    For 2 nodes ``n_i, n_j`` with node features ``f_i, f_j`` and an edge weight of ``w_ij``,
    the gradient on edge ``e_ij`` is given by ``e_ij = w_ij * (f_j - f_i)``.
    If ``X`` is a function of the nodes, then ``matmul(grad_adj, X)`` behaves as the derivative of
    ``X`` in the direction of the gradient of ``features``.
    Parameters
    --------------
        adj: tensor(..., N, N)
            Batches of adjacency matrices of graphs
        features: tensor(..., N)
            Batches of features on the nodes. These features are used to compute the gradient
        normalization: str, Optional
            Normalization strategy for the ``grad_adj`` matrix.
            - 'none': No normalization is applied
            
            - 'row-abs': Normalize such that the absolute sum of each row of ``grad_adj`` is 1.
              All values below the global variable ``EPS=1e-5`` will be set to 0. 
            - 'in-out-field': Normalize such that the sum of the norm of the input and output field
              is 1. The input field is the negative values in the ``grad_adj``, while the
              output field is the positive values. The norm is computed as the root of the squared sum.
              All values below the global variable ``EPS=1e-5`` will be set to 0. 
              For example, if there are no positive values above EPS for a specific node, then
              the output field of that specific node will be set to 0.
            (Default='none')
        
        add_diag: bool, Optional
            Whether to add a diagonal element such as each row of ``grad_adj`` sums to 0.
            The diagonal is added after the normalization.
            
            - If True, ``matmul(grad_adj, X)`` will behave like a forward/backward derivative
              at the local minima/maxima from the features function.
            - If False, ``matmul(grad_adj, X)`` will behave similarily to a center derivative
              with a zero-padding added before the local minima/maxima
            (Default=True)
        absolute_adj: bool, Optional
            Return the absolute value of ``grad_adj``. 
            ``matmul(grad_adj, X)`` becomes a directional smoothing instead of a directional
            derivative.
            (Default=False)
    Returns
    -------------
        grad_adj: tensor(..., N, N)
            Batches of adjacency matrices representing the gradient of the features on the graph.
            
    """
    device = adj.device
    # Compute the gradient of the features
    grad_adj = (adj * (features.unsqueeze(-2) - features.unsqueeze(-1) + EPS))

    # Don't normalize
    if (normalization is None) or (normalization.lower() == 'none'):
        pass

    # Normalize by the sum of each row
    elif normalization.lower() == 'row-abs':
        grad_norm = torch.abs(grad_adj)
        grad_norm *= grad_norm > EPS
        grad_adj = grad_adj / (torch.sum(grad_norm, dim=-1, keepdims=True) + EPS)

    # Normalize by the input and output vector field
    elif normalization.lower() == 'in-out-field':
        pos_adj = grad_adj * (grad_adj > EPS)
        neg_adj = grad_adj * (grad_adj < -EPS)
        out_field = torch.sqrt(torch.sum(pos_adj ** 2, dim=-1, keepdims=True)) + EPS
        in_field = torch.sqrt(torch.sum(neg_adj ** 2, dim=-1, keepdims=True)) + EPS
        in_out_field = out_field + in_field
        grad_adj = (pos_adj / in_out_field) + (neg_adj / in_out_field)

    else:
        raise ValueError('Unsupported normalization option `{}`')

    # Add an element on the diagonal to make each row sum to 0
    if add_diag:
        eye = torch.eye(adj.shape[-1]).to(device)[(None, )*(adj.ndim-2)] # Unsqueeze multiple times
        grad_adj = grad_adj - (eye * torch.sum(grad_adj, dim=-1, keepdims=True))

    # Compute the absolute value
    if absolute_adj:
        grad_adj = torch.abs(grad_adj)

    return grad_adj


def get_eig_adjacency(adj, eig_idx, normalization='none', add_diag=True, absolute_adj=False, normalize_L=False, eig_acos=True):
    r"""
    Computes an adjacency matrix from the gradient of the eigenvectors of the graph Laplacian,
    with the graph being described by the adjacency matrix ``adj``. 
    The gradient is a function on the edges, computed from these node features.
    For 2 nodes ``n_i, n_j`` with node features ``f_i, f_j`` and an edge weight of ``w_ij``,
    the gradient on edge ``e_ij`` is given by ``e_ij = w_ij * (f_j - f_i)``.
    If ``X`` is a function of the nodes, then ``matmul(grad_adj, X)`` behaves as the derivative of
    ``X`` in the direction of the gradient of ``features``.
    Parameters
    --------------
        adj: tensor(..., N, N)
            Batches of adjacency matrices of graphs. It is used to compute the Laplacian matrix,
            and the eigenvectors of the Laplacian.
        eig_idx: int, iterator(int)
            Indexes of the eigenvectors to compute, sorted by smallest eigenvalues. 
            The current function is efficient for low indexes.
            e.g. ``eig_idx=[1, 2, 4]`` will compute the eigenvectors ``[0, 1, 2, 3, 4]``, but will
            then ignore the eigenvectors 0 and 3.
            If ``0`` is in the ``eig_idx``, then the normalized adjacency matrix is returned for
            that specific index, no matter the normalization chosen.
        normalization: str, Optional
            Normalization strategy for the ``grad_adj`` matrix. It does not apply to eig_idx==0.
            - 'none': No normalization is applied
            
            - 'row-abs': Normalize such that the absolute sum of each row of ``grad_adj`` is 1.
              All values below the global variable ``EPS=1e-5`` will be set to 0. 
            - 'in-out-field': Normalize such that the the norm of the input and output field
              are 1. The input field is the negative values in the ``grad_adj``, while the
              output field is the positive values. The norm is computed as the root of the squared sum.
              All values below the global variable ``EPS=1e-5`` will be set to 0. 
              For example, if there are no positive values above EPS for a specific node, then
              the output field of that specific node will be set to 0.
            (Default='none')
        
        add_diag: bool, Optional
            Whether to add a diagonal element such as each row of ``grad_adj`` sums to 0.
            The diagonal is added after the normalization. It does not apply to eig_idx==0.
            
            - If True, ``matmul(grad_adj, X)`` will behave like a forward/backward derivative
              at the local minima/maxima from the features function.
            - If False, ``matmul(grad_adj, X)`` will behave similarily to a center derivative
              with a zero-padding added before the local minima/maxima
            (Default=True)
        absolute_adj: bool, Optional
            Return the absolute value of ``grad_adj``. 
            ``matmul(grad_adj, X)`` becomes a directional smoothing instead of a directional
            derivative.
            (Default=False)
        normalize_L: bool, Optional
            Whether to normalize the Laplacian matrix
            If `False`, then `L = D - A`
            If `True`, then `L = D^-1 (D - A)`
            (Default=False)
        
        eig_acos: bool, Optional
            Whether to compute the arcosine of the eigenvectors instead of the eigenvector.
            This will sort-of `linearize` the eigenvector. If ``normalize=='in-out-field'``,
            this parameter shouldn't change anything, appart from numerical error. For other
            normalization, it will have an important impact, especially near the borders
            of the graph.
    Returns
    -------------
        eig_adj: dict(tensor(..., N, N))
            Dictionary of Batches of adjacency matrices representing the gradient of
            the eigenvectors on the graph. Each key corresponds to a specific ``eig_idx``,
            and the value corresponds to the associated batch of eigen-adjacencies.
            
    """

    # Convert the eig_idx into a list
    try:
        eig_idx = list(eig_idx)
    except:
        eig_idx = [eig_idx]
    
    # Generate an adjacency matrix for each eigenvectors from `eig_idx`
    eigvec = get_k_lowest_eig(adj, max(eig_idx)+1, normalize_L=normalize_L)
    eig_adj = {}
    for ii in eig_idx:
        if ii != 0:
            this_eigvec = eigvec[..., ii]
            if eig_acos:
                this_eigvec = torch.acos(this_eigvec / torch.max(torch.abs(this_eigvec)))
            w_adj = get_adjacency_from_gradient_of_features(adj, features=this_eigvec, 
                        normalization=normalization, add_diag=add_diag, absolute_adj=absolute_adj) 
            eig_adj[ii] = w_adj

        else:
            eig_adj[ii] = adj / (torch.sum(adj.abs(), dim=-1, keepdims=True) + EPS)

    return eig_adj


def aggregate_sum(X, adj, self_loop=False, device='cpu', avg_d=None):
    """
    Aggregate each node by the sum of it's neighbours, weighted by the edge between the nodes.
    Parameters
    --------------
        X: tensor(..., N, N, Din)
            Input feature tensor to aggregate.
        adj: tensor(..., N, N)
            Batches of adjacency matrices of graphs.
    Returns
    -------------
        X_sum: tensor(..., N, N, Din)
            Aggregation results
            
    """

    if self_loop:  # add self connections
        N = adj.shape[-1]
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    X_sum = torch.sum(torch.mul(X, adj.unsqueeze(-1)), dim=-2)
    return X_sum


def aggregate_eigs(X, adj, eig_idx, normalization='none', add_diag=True, agg_type='derivative', normalize_L=False,
                    eig_acos=True, self_loop=False, device='cpu', avg_d=None):
    r"""
    Aggregate the features ``X`` in the direction of the gradient of the Laplacian 
    eigenvectors ``eig_idx`` such as the indexes are sorted by lowest eigenvalues.
    Parameters
    --------------
        X: tensor(..., N, N, Din)
            Input feature tensor to aggregate.
        adj: tensor(..., N, N)
            Batches of adjacency matrices of graphs. It is used to compute the Laplacian matrix,
            and the eigenvectors of the Laplacian.
        eig_idx: int, iterator(int)
            Indexes of the eigenvectors to compute, sorted by smallest eigenvalues. 
            The current function is efficient for low indexes.
            e.g. ``eig_idx=[1, 2, 4]`` will compute the eigenvectors ``[0, 1, 2, 3, 4]``, but will
            then ignore the eigenvectors 0 and 3. 
            If ``0`` is in the ``eig_idx``, then a mean aggregation is done for that specific index.
        normalization: str, Optional
            Normalization strategy for the ``grad_adj`` matrix.
            - 'none': No normalization is applied
            
            - 'row-abs': Normalize such that the absolute sum of each row of ``grad_adj`` is 1.
              All values below the global variable ``EPS=1e-5`` will be set to 0. 
            - 'in-out-field': Normalize such that the the norm of the input and output field
              are 1. The input field is the negative values in the ``grad_adj``, while the
              output field is the positive values. The norm is computed as the root of the squared sum.
              All values below the global variable ``EPS=1e-5`` will be set to 0. 
              For example, if there are no positive values above EPS for a specific node, then
              the output field of that specific node will be set to 0.
            (Default='none')
        
        add_diag: bool, Optional
            Whether to add a diagonal element such as each row of ``grad_adj`` sums to 0.
            The diagonal is added after the normalization.
            
            - If True, ``matmul(grad_adj, X)`` will behave like a forward/backward derivative
              at the local minima/maxima from the features function.
            - If False, ``matmul(grad_adj, X)`` will behave similarily to a center derivative
              with a zero-padding added before the local minima/maxima
            (Default=True)
        agg_type: str, Optional
            Aggregation options
            - 'derivative': Use the eigen-adjacency to compute the directional derivative
              of the signal, then compute the absolute value of the derivative.
            - 'smoothing': Use the absolute value of the eigen-adjacency to compute the
              directional smoothing of the signal.
            - 'both': Use both methods.
            
            (Default='derivative')
        normalize_L: bool, Optional
            Whether to normalize the Laplacian matrix
            If `False`, then `L = D - A`
            If `True`, then `L = D^-1 (D - A)`
            (Default=False)
        
        eig_acos: bool, Optional
            Whether to compute the arcosine of the eigenvectors instead of the eigenvector.
            This will sort-of `linearize` the eigenvector. If ``normalize=='in-out-field'``,
            this parameter shouldn't change anything, appart from numerical error. For other
            normalization, it will have an important impact, especially near the borders
            of the graph.
        self_loop: bool, Optional
            Whether to add a self-loop in the aggregation
            (Default=False)
        device:
            Torch device used to run the computation
            (Default='cpu')
        avg_d: Ignored
            Ignored parameter. Here for compatibility
    Returns
    -------------
        X_agg: tensor(..., N, N, len(eig_idx)*Din) or tensor(..., N, N, len(eig_idx)*2*Din)
            Aggregation results for each eigenvector, concatenated alongside the last dimension
            The last dimension is of size ``len(eig_idx)*2*Din`` when ``agg_type=='both'``.
            
    """

    agg_type = agg_type.lower()
    if agg_type not in ['derivative', 'smoothing', 'both']:
        raise ValueError(f'Unknown agg_type "{agg_type}"')

    # Get dictionary of adjacency matrices
    adj_dict = get_eig_adjacency(adj, eig_idx, normalization=normalization, add_diag=add_diag, 
                                absolute_adj=False, normalize_L=normalize_L, eig_acos=eig_acos)

    X_agg = []
    for ii, this_adj in adj_dict.items():
        # Compute derivative
        if ((agg_type == 'derivative') or (agg_type == 'both')) and (ii != 0):
            this_agg = aggregate_sum(X, this_adj, self_loop=self_loop, device=device)
            X_agg.append(this_agg)

        # Compute smoothing
        if (agg_type == 'smoothing') or (agg_type == 'both') or (ii == 0):
            this_agg = aggregate_sum(X, this_adj.abs(), self_loop=self_loop, device=device)
            X_agg.append(this_agg)
        
    return torch.cat(X_agg, dim=-1)
