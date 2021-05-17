# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:56:16 2021

@author: eliphat
"""
import random
import argparse
import contextlib
import torch
import torch.optim as optim
from merger.data_flower import all_h5
from merger.merger_net import Net
from merger.composed_chamfer import composed_sqrt_chamfer


arg_parser = argparse.ArgumentParser(description="Training Skeleton Merger. Valid .h5 files must contain a 'data' array of shape (N, n, 3) and a 'label' array of shape (N, 1).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-t', '--train-data-dir', type=str, default='../point_cloud/train',
                        help='Directory that contains training .h5 files.')
arg_parser.add_argument('-v', '--val-data-dir', type=str, default='../point_cloud/val',
                        help='Directory that contains validation .h5 files.')
arg_parser.add_argument('-c', '--subclass', type=int, default=14,
                        help='Subclass label ID to train on.')  # 14 is `chair` class.
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='merger.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Pytorch device for training.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-e', '--epochs', type=int, default=80,
                        help='Number of epochs to train.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')


def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))


def feed(net, optimizer, x_set, train, shuffle, batch, epoch):
    running_loss = 0.0
    running_lrc = 0.0
    running_ldiv = 0.0
    net.train(train)
    if shuffle:
        x_set = list(x_set)
        random.shuffle(x_set)
    with contextlib.suppress() if train else torch.no_grad():
        for i in range(len(x_set) // batch):
            idx = slice(i * batch, (i + 1) * batch)
            refp = next(net.parameters())
            batch_x = torch.tensor(x_set[idx], device=refp.device)
            if train:
                optimizer.zero_grad()
            RPCD, KPCD, KPA, LF, MA = net(batch_x)
            blrc = composed_sqrt_chamfer(batch_x, RPCD, MA)
            bldiv = L2(LF)
            loss = blrc + bldiv
            if train:
                loss.backward()
                optimizer.step()
    
            # print statistics
            running_lrc += blrc.item()
            running_ldiv += bldiv.item()
            running_loss += loss.item()
            print('[%s%d, %4d] loss: %.4f Lrc: %.4f Ldiv: %.4f' %
                  ('VT'[train], epoch, i, running_loss / (i + 1), running_lrc / (i + 1), running_ldiv / (i + 1)))
    return running_loss / (i + 1), running_lrc / (i + 1), running_ldiv / (i + 1)


if __name__ == '__main__':
    ns = arg_parser.parse_args()
    DATASET = ns.train_data_dir
    TESTSET = ns.val_data_dir
    batch = ns.batch
    x, xl = all_h5(DATASET, True, True, subclasses=(ns.subclass,), sample=None)  # n x 2048 x 3
    x_test, xl_test = all_h5(TESTSET, True, True, subclasses=(ns.subclass,), sample=None)
    net = Net(ns.max_points, ns.n_keypoint).to(ns.device)
    optimizer = optim.Adadelta(net.parameters(), eps=1e-2)
    for epoch in range(ns.epochs):
        feed(net, optimizer, x, True, True, batch, epoch)
        feed(net, optimizer, x_test, False, False, batch, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
        }, ns.checkpoint_path)
