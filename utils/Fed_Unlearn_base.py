# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:29:20 2020

@author: user
"""
import torch
import copy
from models.Fed import FedAvg
from models.resnet import ResNet18
from models.Update import LocalUpdate
import numpy as np
def global_train_once_params(global_model_params, train_datasets, FL_params):
    forget_client_idx = 0
    if FL_params.arch == 'resnet':
        initial_model = ResNet18(num_classes=FL_params.num_classes, norm_type=FL_params.norm_type)
    else:
        raise Exception('Unknown arch')

    model_dict = copy.deepcopy(global_model_params)
    all_global_model_dicts = []

    idxs_users = [i for i in range(0, FL_params.num_users)]
    quality = FL_params.local_epoch
    # print("idx_users", idxs_users)

    current_model_state_dict = copy.deepcopy(model_dict)
    current_model = copy.deepcopy(initial_model)
    current_model.load_state_dict(current_model_state_dict)

    party_models_state = []
    party_losses = []

    for idx in idxs_users:
        local_dataset = train_datasets[idx]
        local = LocalUpdate(args=FL_params, dataset=local_dataset, loss_global=FL_params.loss_avg, quality=quality,
                            idx=idx)
        w, loss, loss_diff = local.train(net=copy.deepcopy(current_model).to(FL_params.device))
        party_models_state.append(copy.deepcopy(w))
        party_losses.append(loss)

    party_models_state.pop(forget_client_idx)

    return party_models_state

def unlearning(old_GMs_params, old_CMs_params, train_datasets,  FL_params):

    # Preprocess the old models' parameters
    old_global_models_params = copy.deepcopy(old_GMs_params)
    old_client_models_params = copy.deepcopy(old_CMs_params)

    forget_client = 0
    for ii in range(FL_params.epochs):
        temp = old_client_models_params[ii * FL_params.num_users: ii * FL_params.num_users + FL_params.num_users]
        temp.pop(forget_client)
        old_client_models_params.append(temp)
    old_client_models_params = old_client_models_params[-FL_params.epochs:]

    # selected_GMs_params = old_global_models_params
    # selected_CMs_params = old_client_models_params
    GM_intv = np.arange(0, FL_params.epochs+1, FL_params.un_itv, dtype=np.int16())
    CM_intv = GM_intv - 1
    CM_intv = CM_intv[1:]
    print("Max index in GM_intv:", max(GM_intv))
    print("Length of old_GMs_params:", len(old_GMs_params))

    selected_GMs_params= [old_global_models_params[ii] for ii in GM_intv]
    selected_CMs_params = [old_client_models_params[jj] for jj in CM_intv]
    epoch = 0
    unlearn_global_models_params = list()
    unlearn_global_models_params.append(copy.deepcopy(selected_GMs_params[0]))

    print("Type of selected_CMs_params[epoch]:", type(selected_CMs_params[epoch]))

    # 如果 selected_CMs_params[epoch] 是列表，则打印其长度
    if isinstance(selected_CMs_params[epoch], list):
        print("Length of selected_CMs_params[epoch]:", len(selected_CMs_params[epoch]))

    new_global_model_params = FedAvg(selected_CMs_params[epoch])  # Assuming FedAvg is adapted to work with model parameters

    unlearn_global_models_params.append(copy.deepcopy(new_global_model_params))
    print("Federated Unlearning Global Epoch = {}".format(epoch))

    CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
    FL_params.local_epoch = np.ceil(FL_params.local_epoch*FL_params.forget_ratio)
    FL_params.local_epoch = np.int16(FL_params.local_epoch)

    CONST_global_epoch = copy.deepcopy(FL_params.epochs)

    FL_params.epochs = CM_intv.shape[0]

    for epoch in range(1, FL_params.epochs):
        print("Federated Unlearning Global Epoch = {}".format(epoch))
        global_model_params = unlearn_global_models_params[epoch]

        # Assuming global_train_once is adapted to work with model parameters
        new_client_models_params = global_train_once_params(global_model_params, train_datasets,
                                                            FL_params)

        new_GM_params = unlearning_step_once_params(selected_CMs_params[epoch], new_client_models_params,
                                                    selected_GMs_params[epoch], global_model_params)

        unlearn_global_models_params.append(new_GM_params)

    FL_params.local_epoch = CONST_local_epoch
    FL_params.epochs = CONST_global_epoch

    return unlearn_global_models_params


def unlearning_step_once_params(old_client_models_params, new_client_models_params, global_model_before_forget_params, global_model_after_forget_params):
    old_param_update = dict()
    new_param_update = dict()

    new_global_model_state = copy.deepcopy(global_model_after_forget_params)
    return_model_state = dict()

    assert len(old_client_models_params) == len(new_client_models_params)

    for layer in global_model_before_forget_params.keys():
        old_param_update[layer] = torch.zeros_like(global_model_before_forget_params[layer])
        new_param_update[layer] = torch.zeros_like(global_model_before_forget_params[layer])
        return_model_state[layer] = torch.zeros_like(global_model_before_forget_params[layer])

        for ii in range(len(new_client_models_params)):
            old_param_update[layer] += old_client_models_params[ii][layer]
            new_param_update[layer] += new_client_models_params[ii][layer]

        old_param_update[layer] /= (ii + 1)
        new_param_update[layer] /= (ii + 1)

        old_param_update[layer] -= global_model_before_forget_params[layer]
        new_param_update[layer] -= global_model_after_forget_params[layer]

        step_length = torch.norm(old_param_update[layer])
        step_direction = new_param_update[layer] / torch.norm(new_param_update[layer])

        return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction

    return return_model_state



































