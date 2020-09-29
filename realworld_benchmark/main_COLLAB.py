
"""
    IMPORTING LIBS
"""

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self



"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from data.COLLAB import COLLABDataset
from nets.COLLAB_edge_classification.eig_net import EIGNet
from train.train_COLLAB_edge_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network



"""
    GPU Setup
"""

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:' ,torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device






"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = EIGNet(net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    t0 = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name

    # assert net_params['self_loop'] == False, "No self-loop support for %s dataset" % DATASET_NAME

    if MODEL_NAME in ['GatedGCN']:
        if net_params['pos_enc']:
            print("[!] Adding graph positional encoding" ,net_params['pos_enc_dim'])
            dataset._add_positional_encodings(net_params['pos_enc_dim'])
            print('Time PE:' ,time.time( ) -t0)

    graph = dataset.graph

    evaluator = dataset.evaluator

    train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg = dataset.train_edges, dataset.val_edges, dataset.val_edges_neg, dataset.test_edges, dataset.test_edges_neg

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format
            (DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Graph: ", graph)
    print("Training Edges: ", len(train_edges))
    print("Validation Edges: ", len(val_edges) + len(val_edges_neg))
    print("Test Edges: ", len(test_edges) + len(test_edges_neg))

    model = EIGNet(net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'])

    epoch_train_losses = []
    epoch_train_hits, epoch_val_hits, epoch_test_hits = [], [], []


    start_epoch = 0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(start_epoch, params['epochs']), mininterval=params['print_epoch_interval'],
                  maxinterval=None, unit='epoch', initial=start_epoch, total=params['epochs']) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, optimizer = train_epoch(model, optimizer, device, graph, train_edges,
                                                          params['batch_size'], net_params['augmentation'], epoch)

                epoch_train_hit, epoch_val_hit, epoch_test_hit = evaluate_network(
                    model, device, graph, train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg, evaluator,
                    params['batch_size'], epoch)

                epoch_train_losses.append(epoch_train_loss)
                epoch_train_hits.append(epoch_train_hit)
                epoch_val_hits.append(epoch_val_hit)
                epoch_test_hits.append(epoch_test_hit)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)

                writer.add_scalar('train/_hits@10', epoch_train_hit[0] * 100, epoch)
                writer.add_scalar('train/_hits@50', epoch_train_hit[1] * 100, epoch)
                writer.add_scalar('train/_hits@100', epoch_train_hit[2] * 100, epoch)

                writer.add_scalar('val/_hits@10', epoch_val_hit[0] * 100, epoch)
                writer.add_scalar('val/_hits@50', epoch_val_hit[1] * 100, epoch)
                writer.add_scalar('val/_hits@100', epoch_val_hit[2] * 100, epoch)

                writer.add_scalar('test/_hits@10', epoch_test_hit[0] * 100, epoch)
                writer.add_scalar('test/_hits@50', epoch_test_hit[1] * 100, epoch)
                writer.add_scalar('test/_hits@100', epoch_test_hit[2] * 100, epoch)

                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, train_hits=epoch_train_hit[1],
                              val_hits=epoch_val_hit[1], test_hits=epoch_test_hit[1])

                per_epoch_time.append(time.time() - start)


                scheduler.step(epoch_val_hit[1])

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break


    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    train_hit, val_hit, test_hit = evaluate_network(
        model, device, graph, train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg, evaluator,
        params['batch_size'], epoch)

    print(
        f"Test:\nHits@10: {test_hit[0] * 100:.4f}% \nHits@50: {test_hit[1] * 100:.4f}% \nHits@100: {test_hit[2] * 100:.4f}% \n")
    print(
        f"Val:\nHits@10: {val_hit[0] * 100:.4f}% \nHits@50: {val_hit[1] * 100:.4f}% \nHits@100: {val_hit[2] * 100:.4f}% \n")
    print(
        f"Train:\nHits@10: {train_hit[0] * 100:.4f}% \nHits@50: {train_hit[1] * 100:.4f}% \nHits@100: {train_hit[2] * 100:.4f}% \n")
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()


def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--graph_norm', help="Please give a value for graph_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--expid', help='Experiment id.')
    parser.add_argument('--type_net', default='simple', help='Type of net')
    parser.add_argument('--lap_norm', default='none', help='Laplacian normalisation')
    parser.add_argument('--augmentation', type=float, default=0., help='Dynamically augmenting with rotations, angle in degrees')


    # eig params
    parser.add_argument('--aggregators', type=str, help='Aggregators to use.')
    parser.add_argument('--scalers', type=str, help='Scalers to use.')
    parser.add_argument('--towers', type=int, default=5, help='Towers to use.')
    parser.add_argument('--divide_input_first', type=bool, help='Whether to divide the input in first layer.')
    parser.add_argument('--divide_input_last', type=bool, help='Whether to divide the input in last layers.')
    parser.add_argument('--edge_dim', type=int, help='Size of edge embeddings.')
    parser.add_argument('--pretrans_layers', type=int, help='pretrans_layers.')
    parser.add_argument('--posttrans_layers', type=int, help='posttrans_layers.')

    args = parser.parse_args()


    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = COLLABDataset(DATASET_NAME, norm=args.lap_norm)
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)


    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.graph_norm is not None:
        net_params['graph_norm'] = True if args.graph_norm == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.aggregators is not None:
        net_params['aggregators'] = args.aggregators
    if args.scalers is not None:
        net_params['scalers'] = args.scalers
    if args.towers is not None:
        net_params['towers'] = args.towers
    if args.divide_input_first is not None:
        net_params['divide_input_first'] = args.divide_input_first
    if args.divide_input_last is not None:
        net_params['divide_input_last'] = args.divide_input_last
    if args.edge_dim is not None:
        net_params['edge_dim'] = args.edge_dim
    if args.pretrans_layers is not None:
        net_params['pretrans_layers'] = args.pretrans_layers
    if args.posttrans_layers is not None:
        net_params['posttrans_layers'] = args.posttrans_layers
    if args.type_net is not None:
        net_params['type_net'] = args.type_net
    if args.augmentation is not None:
        net_params['augmentation'] = args.augmentation

    # COLLAB
    net_params['in_dim'] = dataset.graph.ndata['feat'].shape[-1]
    net_params['in_dim_edge'] = dataset.graph.edata['feat'].shape[-1]
    net_params['n_classes'] = 1  # binary prediction

    D = torch.sparse.sum(dataset.graph.adjacency_matrix(transpose=True), dim=-1).to_dense()
    net_params['avg_d'] = dict(lin=torch.mean(D),
                                   exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                                   log=torch.mean(torch.log(D + 1)))

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)


main()