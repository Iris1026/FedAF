#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
# def FedAvg(w, first_client_weight=2.0):
#     """
#     联邦平均算法，增加第一个客户端的权重。
#
#     :param w: 客户端模型参数的列表。
#     :param first_client_weight: 第一个客户端的权重因子。
#     :return: 联邦平均后的模型参数。
#     """
#     w_avg = copy.deepcopy(w[0])
#     total_weight = first_client_weight
#
#     # 首先处理第一个客户端的权重
#     for k in w_avg.keys():
#         w_avg[k] *= first_client_weight
#
#     # 处理其他客户端
#     for i in range(1, len(w)):
#         for k in w_avg.keys():
#             w_avg[k] += w[i][k]
#         total_weight += 1
#
#     # 将总权重归一化
#     for k in w_avg.keys():
#         w_avg[k] = torch.div(w_avg[k], total_weight)
#
#     return w_avg