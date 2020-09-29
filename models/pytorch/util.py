from __future__ import division
from __future__ import print_function

import pickle
import torch
import torch.nn.functional as F


def load_dataset(data_path, loss, only_nodes, only_graph, print_baseline=True, eigs=False):
    with open(data_path, 'rb') as f:
        if eigs:
            (adj, features, node_labels, graph_labels, graph_eigs) = pickle.load(f)
        else:
            (adj, features, node_labels, graph_labels) = pickle.load(f)

        if only_graph:
            max_graph_labels = torch.cat([gls.max(0)[0].unsqueeze(0) for gls in graph_labels['train']]).max(0)[0]

            for dset in graph_labels.keys():
                graph_labels[dset] = [gls / max_graph_labels for gls in graph_labels[dset]]

            if print_baseline:
                # calculate baseline
                mean_graph_labels = torch.cat([gls.mean(0).unsqueeze(0) for gls in graph_labels['train']]).mean(0)

                for dset in graph_labels.keys():
                    if dset not in ['train', 'val']:
                        baseline_graph = [mean_graph_labels.repeat([gls.shape[0], 1]) for gls in graph_labels[dset]]

                        print("Baseline loss ", dset,
                              specific_loss_multiple_batches((baseline_graph),
                                                             (graph_labels[dset]),
                                                             loss=loss, only_nodes=only_nodes, only_graph=only_graph))

        elif only_nodes:
            max_node_labels = torch.cat([nls.max(0)[0].max(0)[0].unsqueeze(0) for nls in node_labels['train']]).max(0)[
                0]

            for dset in node_labels.keys():
                node_labels[dset] = [nls / max_node_labels for nls in node_labels[dset]]

            if print_baseline:
                # calculate baseline
                mean_node_labels = torch.cat([nls.mean(0).mean(0).unsqueeze(0) for nls in node_labels['train']]).mean(0)

                for dset in node_labels.keys():
                    if dset not in ['train', 'val']:
                        baseline_nodes = [mean_node_labels.repeat(list(nls.shape[0:-1]) + [1]) for nls in
                                          node_labels[dset]]

                        print("Baseline loss ", dset,
                              specific_loss_multiple_batches((baseline_nodes),
                                                             (node_labels[dset]),
                                                             loss=loss, only_nodes=only_nodes, only_graph=only_graph))


        else:
            max_node_labels = torch.cat([nls.max(0)[0].max(0)[0].unsqueeze(0) for nls in node_labels['train']]).max(0)[
                0]
            max_graph_labels = torch.cat([gls.max(0)[0].unsqueeze(0) for gls in graph_labels['train']]).max(0)[0]

            for dset in node_labels.keys():
                node_labels[dset] = [nls / max_node_labels for nls in node_labels[dset]]
                graph_labels[dset] = [gls / max_graph_labels for gls in graph_labels[dset]]

            if print_baseline:
                # calculate baseline
                mean_node_labels = torch.cat([nls.mean(0).mean(0).unsqueeze(0) for nls in node_labels['train']]).mean(0)
                mean_graph_labels = torch.cat([gls.mean(0).unsqueeze(0) for gls in graph_labels['train']]).mean(0)

                for dset in node_labels.keys():
                    if dset not in ['train', 'val']:
                        baseline_nodes = [mean_node_labels.repeat(list(nls.shape[0:-1]) + [1]) for nls in
                                          node_labels[dset]]
                        baseline_graph = [mean_graph_labels.repeat([gls.shape[0], 1]) for gls in graph_labels[dset]]

                        print("Baseline loss ", dset,
                              specific_loss_multiple_batches((baseline_nodes, baseline_graph),
                                                             (node_labels[dset], graph_labels[dset]),
                                                             loss=loss, only_nodes=only_nodes, only_graph=only_graph))

        if eigs:
            return adj, features, node_labels, graph_labels, graph_eigs
        else:
            return adj, features, node_labels, graph_labels




SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()


def get_loss(loss, output, target):
    if loss == "mse":
        return F.mse_loss(output, target)
    elif loss == "cross_entropy":
        if len(output.shape) > 2:
            (B, N, _) = output.shape
            output = output.reshape((B * N, -1))
            target = target.reshape((B * N, -1))
        _, target = target.max(dim=1)
        return F.cross_entropy(output, target)
    else:
        print("Error: loss function not supported")


def total_loss(output, target, loss='mse', only_nodes=False, only_graph=False):
    """ returns the average of the average losses of each task """
    assert not (only_nodes and only_graph)

    if only_nodes:
        nodes_loss = get_loss(loss, output[0], target[0])
        return nodes_loss
    elif only_graph:
        graph_loss = get_loss(loss, output[1], target[1])
        return graph_loss
    else:
        nodes_loss = get_loss(loss, output[0], target[0])
        graph_loss = get_loss(loss, output[1], target[1])
        weighted_average = (nodes_loss * output[0].shape[-1] + graph_loss * output[1].shape[-1]) / (
                output[0].shape[-1] + output[1].shape[-1])
        return weighted_average


def total_loss_multiple_batches(output, target, loss='mse', only_nodes=False, only_graph=False):
    """ returns the average of the average losses of each task over all batches,
        batches are weighted equally regardless of their cardinality or graph size """
    return sum([total_loss((output[0][batch], output[1][batch]), (target[0][batch], target[1][batch]),
                           loss, only_nodes, only_graph).data.item() for batch in range(len(output[0]))]) / len(output[0])


def specific_loss(output, target, loss='mse', only_nodes=False, only_graph=False):
    """ returns the average loss for each task """
    assert not (only_nodes and only_graph)

    if only_nodes:
        nodes_losses = [get_loss(loss, output[0][:, k], target[0][:, k]).item() for k in
                        range(output[0].shape[-1])]
        return nodes_losses
    elif only_graph:
        graph_loss = [get_loss(loss, output[1][:, k], target[1][:, k]).item() for k in range(output[1].shape[-1])]
        return graph_loss

    else:
        nodes_losses = [get_loss(loss, output[0][:, :, k], target[0][:, :, k]).item() for k in range(output[0].shape[-1])]
        graph_loss = [get_loss(loss, output[1][:, k], target[1][:, k]).item() for k in range(output[1].shape[-1])]
        return nodes_losses + graph_loss


def specific_loss_multiple_batches(output, target, loss='mse', only_nodes=False, only_graph=False):
    """ returns the average loss over all batches for each task,
        batches are weighted equally regardless of their cardinality or graph size """
    assert not (only_nodes and only_graph)

    n_batches = len(output[0])
    classes = (output[0][0].shape[-1] if not only_graph else 0) + (output[1][0].shape[-1] if not only_nodes else 0)

    sum_losses = [0] * classes
    for batch in range(n_batches):
        spec_loss = specific_loss((output[0][batch], output[1][batch]), (target[0][batch], target[1][batch]), loss,
                                  only_nodes, only_graph)
        for par in range(classes):
            sum_losses[par] += spec_loss[par]

    return [sum_loss / n_batches for sum_loss in sum_losses]
