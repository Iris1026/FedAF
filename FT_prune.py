import torch
import copy
import numpy as np
from utils.sampling import cifar_iid,mnist_iid,Data_split, remove_class_datasets
from utils.options import args_parser
from utils.poisoning import mark_class_erroneous_samples, mark_class_erroneous_samples_test, CIFAR10_class_BackdoorOnly, CIFAR10_class_Backdoor
from models.Fed import FedAvg
from models.test import test_img
from models.Update import LocalUpdate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from models.resnet import ResNet18
from class_pruner import acculumate_feature, calculate_cp, \
    get_threshold_by_sparsity, TFIDFPruner

def initialize_model(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model

def rest_class(dataset, class_to_remove):
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label != class_to_remove:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)

def remove_class(dataset, class_to_remove):
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == class_to_remove:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)


if __name__ == '__main__':
    # seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # parse args
    args = args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=trans_mnist, download=True)
        test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=trans_mnist, download=True)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        test_dataset = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
    else:
        print("Can not find dataset!")

    if args.dataset=='cifar':
        dict_users = cifar_iid(train_dataset, args.num_users)
    elif args.dataset == 'mnist':
        dict_users = mnist_iid(train_dataset, args.num_users)

    train_dataloader = DataLoader( train_dataset, batch_size=128, shuffle=True)
    class_to_remove = 5
    remove_train_dataset = remove_class(train_dataset, class_to_remove)
    remove_train_dataloader = DataLoader(remove_train_dataset, batch_size=128, shuffle=True)
    rest_train_dataset = rest_class(train_dataset, class_to_remove)
    rest_train_dataloader = DataLoader(rest_train_dataset, batch_size=128, shuffle=True)

    train_datasets, trainloader_lst = Data_split(train_dataset, dict_users, args)
    removed_train_datasets = remove_class_datasets(train_datasets, class_to_remove=5)
    removed_train_dataloaders  = [DataLoader(dataset, batch_size=128, shuffle=True) for dataset in
                       removed_train_datasets]

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    target_class = 5
    new_label = 9
    trigger_size = 2

    if args.arch == 'resnet':
        initial_model = ResNet18(num_classes=args.num_classes, norm_type=args.norm_type).to(args.device)
    else:
        raise Exception('Unknown arch')

    loss_avg = args.loss_avg
    all_global_model_dicts = []
    all_party_model_dicts = []

    model_dict = copy.deepcopy(initial_model.state_dict())
    all_global_model_dicts.append(copy.deepcopy(initial_model.state_dict()))
    optimal_params = {}
    fisher_information = {}
    ewc_enabled = False

    for iter in range(args.epochs):

        print(f"\nFederated Learning Round: {iter}")

        idxs_users = [i for i in range(0, args.num_users)]

        quality = args.local_epoch
        print("idx_users", idxs_users)

        current_model_state_dict = copy.deepcopy(model_dict)
        current_model = copy.deepcopy(initial_model)
        current_model.load_state_dict(current_model_state_dict)

        party_models_state = []
        party_losses = []

        for idx in idxs_users:
            if iter < (args.epochs - args.ft_epochs):
                local_dataset = train_datasets[idx]
                net = copy.deepcopy(current_model).to(args.device)
                local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality,
                                    idx=idx)
                w, loss, loss_diff = local.train(net)

            else:
                '''pre-processing'''
                net = copy.deepcopy(current_model).to(args.device)
                feature_iit, classes = acculumate_feature(net, train_dataloader, 20)
                tf_idf_map = calculate_cp(feature_iit, classes, args.dataset, 0,
                                          unlearn_class=5)
                threshold = get_threshold_by_sparsity(tf_idf_map, 0.05)
                #print('threshold', threshold)
                '''pruning'''
                cp_config = {"threshold": threshold, "map": tf_idf_map}
                config_list = [{
                    'sparsity': 0.05,
                    'op_types': ['Conv2d']
                }]
                pruner = TFIDFPruner(net, config_list, cp_config=cp_config)
                pruner.compress()
                local_dataset = removed_train_datasets[idx]
                local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality,
                                    idx=idx)
                w, loss, loss_diff = local.train(net)
            party_models_state.append(copy.deepcopy(w))
            party_losses.append(loss)

        current_model_state_dict = FedAvg(party_models_state)
        model_dict = copy.deepcopy(current_model_state_dict)

        eval_glob = copy.deepcopy(net).to(args.device)
        eval_glob.load_state_dict(model_dict)
        eval_glob.eval()
        acc1, loss_test1 = test_img(eval_glob, test_dataset, args)
        print(f'FedAvg global Accuracy of test_dataset , round {iter} = {acc1}')
        print(f'FedAvg global Loss of test_dataset , round {iter} = {loss_test1}')
        acc2, loss_test2 = test_img(eval_glob, rest_train_dataset , args)
        print(f'FedAvg global Accuracy of rest_dataset , round {iter} = {acc2}')
        print(f'FedAvg global Loss of rest_dataset, round {iter} = {loss_test2}')
        acc3, loss_test3 = test_img(eval_glob, remove_train_dataset , args)
        print(f'FedAvg global Accuracy of unlearn_dataset , round {iter} = {acc3}')
        print(f'FedAvg global Loss of unlearn_dataset, round {iter} = {loss_test3}')


