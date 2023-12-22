#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="global rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_epoch', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--loss_avg', type = float, default=100, help='initiation global loss value')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', action='store_false', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose',  default = False, help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    # poisoning
    parser.add_argument('--percent_poison', type=float, default=1, help="Poisoning rate")
    parser.add_argument('--scale', type=float, default=1, help="scale")

    # unlearning
    parser.add_argument('--num_local_epochs_unlearn', type=int, default=5, help="rounds of local unlearning")
    parser.add_argument('--distance_threshold', type=int, default=200, help="Distance of Reference Model to party0_model")

    parser.add_argument('--action', default='train', choices=['train', 'eval', 'fine-tune'],
                        help='classification experiments (default: train)')

    #
    # parser.add_argument('--dataset', default='cifar10', choices=['cifar10',
    #                                                              'cifar100',
    #                                                              'caltech-101',
    #                                                              'caltech-256',
    #                                                              'imagenet'],
    #                     help='experiment dataset (default: cifar10)')
    parser.add_argument('--arch', default='resnet', choices=['resnet'],
                        help='architecture (default: resnet)')

    parser.add_argument('--norm-type', default='bn', choices=['bn', 'none'],
                        help='norm type (default: bn)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size (default: 64)')

    # watermark
    parser.add_argument('--embed', action='store_true', default=False,
                        help='turn on watermarking')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='threshold for watermarking (default: 0.1)')
    parser.add_argument('--lamda', type=float, default=0.01,
                        help='coe of watermark reg in loss function (default: 0.01)')
    parser.add_argument('--divider', type=int, default=2,
                        help='describe the fraction of elements to be zeros in watermarking (default: 2)')

    # paths
    parser.add_argument('--pretrained-path',
                        help='path of pretrained model')
    parser.add_argument('--lr-config', default='default.json',
                        help='lr config json file')
    #Reconstruction
    parser.add_argument('--forget_ratio', type=float, default=0.8, help="forget_local_epoch_ratio")
    parser.add_argument('--un_itv', type=int, default=1, help="unlearn_interval")

    # FedAF
    parser.add_argument('--ewc_lambda', type=float, default=10, help="ewc_lambda")
    parser.add_argument('--ewc_epochs', type=int, default=1, help="ewc_epochs")
    args = parser.parse_args()
    return args
