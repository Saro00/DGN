# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import random as rd
import torch.nn as nn
import math

from .metrics import accuracy_MNIST_CIFAR as accuracy

def train_epoch(model, optimizer, device, data_loader, epoch, augmentation, flip, distortion):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)

        if augmentation > 1e-7:
            batch_graphs_eig = batch_graphs.ndata['eig'].clone()

            angle = (torch.rand(batch_x[:, 0].shape) - 0.5) * 2 * augmentation
            sine = torch.sin(angle * math.pi / 180)
            batch_graphs.ndata['eig'][:, 1] = torch.mul((1 - sine**2)**(0.5), batch_graphs_eig[:, 1])  \
                                              + torch.mul(sine, batch_graphs_eig[:, 2])
            batch_graphs.ndata['eig'][:, 2] = torch.mul((1 - sine**2) ** (0.5), batch_graphs_eig[:, 2]) \
                                              - torch.mul(sine, batch_graphs_eig[:, 1])
        if flip:
            batch_graphs_eig = batch_graphs.ndata['eig'][:, 2].to(device)
            sign_flip = torch.rand(batch_graphs_eig.size()).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0; sign_flip[sign_flip < 0.5] = -1.0
            batch_graphs.ndata['eig'][:, 2] = torch.mul(sign_flip, batch_graphs_eig)

        if distortion > 1e-7:
            batch_graphs_eig = batch_graphs.ndata['eig'].clone()
            dist = (torch.rand(batch_x[:, 0].shape) - 0.5) * 2 * distortion
            batch_graphs.ndata['eig'][:, 1] = torch.mul(dist, torch.mean(torch.abs(batch_graphs_eig[:, 1]), dim=-1, keepdim=True)) + batch_graphs_eig[:, 1]
            batch_graphs.ndata['eig'][:, 2] = torch.mul(dist, torch.mean(torch.abs(batch_graphs_eig[:, 2]), dim=-1, keepdim=True)) + batch_graphs_eig

        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        if augmentation  > 1e-7 or distortion > 1e-7:
            batch_graphs.ndata['eig'] = batch_graphs_eig.detach()
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc