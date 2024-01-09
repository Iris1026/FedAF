#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset,TensorDataset,Dataset
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))  #不可以取相同的
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    # seeds
    np.random.seed(0)
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1]*len(k_idcs)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    # Calculate weights based on the number of data points for each client
    client_weights = [len(idcs) for idcs in client_idcs]

    return client_idcs


def load_dataset(args):
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=transform, download=True)
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    # ])

    # train_dataset = ImageFolder(train_dir, transform=transform)
    # test_dataset = ImageFolder(test_dir, transform=transform)

    x_train = [train_dataset[i][0] for i in range(len(train_dataset))]
    y_train = [train_dataset[i][1] for i in range(len(train_dataset))]

    x_test = [test_dataset[i][0] for i in range(len(test_dataset))]
    y_test = [test_dataset[i][1] for i in range(len(test_dataset))]

    print("length of x_train", len(x_train))
    print("length of x_test", len(x_test))

    x_train = np.stack([x.numpy() if isinstance(x, torch.Tensor) else x for x in x_train])
    x_test = np.stack([x.numpy() if isinstance(x, torch.Tensor) else x for x in x_test])

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # if args.dataset == 'mnist':
    #     dict_users = mnist_iid(train_dataset, args.num_users)
    #
    # elif args.dataset == 'cifar':
    #     dict_users = cifar_iid(train_dataset, args.num_users)

    # if args.iid:
    #     client_idcs = []
    #     for i in range(args.num_users):
    #         client_idcs.append(list(range(i, len(x_train),args.num_users)))
    #         print("length ofclient_idcs ",len(client_idcs[i]))
    # else:
    #     client_idcs = dirichlet_split_noniid(y_train, 0.8, args.num_users)

    client_idcs = []
    for i in range(args.num_users):
        client_idcs.append(list(range(i, len(x_train),args.num_users)))
        print("length ofclient_idcs ",len(client_idcs[i]))

    return x_train, y_train, x_test, y_test, client_idcs

def Data_split(train_dataset,dict_users, args):
    train_datasets = []
    for i in range(0, args.num_users):
        client_indices = dict_users[i]

        x_train_parties = [train_dataset[idx][0] for idx in client_indices]
        y_train_parties = [train_dataset[idx][1] for idx in client_indices]

        dataset = TensorDataset(torch.stack(x_train_parties), torch.Tensor(y_train_parties).long())
        train_datasets.append(dataset)
    trainloader_lst = [DataLoader(dataset, batch_size=128, shuffle=True) for dataset in
                       train_datasets]

    return train_datasets, trainloader_lst

def generate_debiased_labels(teacher_models, R_loader, target_class,args):
    debiased_labels = []
    with torch.no_grad():
        for data, labels in R_loader:
            data = data.to(args.device)
            predictions = torch.stack([model(data) for model in teacher_models])
            average_predictions = predictions.mean(0)
            sigma_values = torch.ones_like(average_predictions)
            target_preds = average_predictions[:, target_class].unsqueeze(1)
            target_preds_mean = target_preds / average_predictions.mean(dim=1, keepdim=True)
            sigma_values[:, target_class] = target_preds_mean.squeeze()
            debias_vectors = sigma_values * average_predictions
            debiased_predictions = debias_vectors / torch.norm(debias_vectors, p=1, dim=1, keepdim=True)
            _, debiased_labels_batch = torch.max(debiased_predictions, dim=1)
            debiased_labels.append(debiased_labels_batch)

    return torch.cat(debiased_labels, 0)

def generate_bd_R(train_dataset, dict_users, target_class, new_label, args):
    R_datasets = []
    R_loaders = []
    if args.eval == 'backdoor':
        for i in range(0, args.num_users):
            client_indices = dict_users[i]

            x_train_R = [train_dataset[idx][0] for idx in client_indices if train_dataset[idx][1] == target_class]
            y_train_R_tensor = torch.full((len(x_train_R),), new_label, dtype=torch.long)

            if x_train_R:
                R_dataset = TensorDataset(torch.stack(x_train_R), y_train_R_tensor)
                R_datasets.append(R_dataset)
                R_loader = DataLoader(R_dataset, batch_size=args.batch_size, shuffle=True)
                R_loaders.append(R_loader)
    return R_datasets, R_loaders

def generate_EM_R(label_5_dataset, label_5_global_to_local, dict_users,args):
    R_datasets = []
    R_loaders = []
    for i in range(0, args.num_users):
        client_indices = dict_users[i]
        local_indices = [label_5_global_to_local[idx] for idx in client_indices if
                         idx in label_5_global_to_local]
        x_train_R = [label_5_dataset[idx][0] for idx in local_indices]
        y_train_R = [label_5_dataset[idx][1] for idx in local_indices]

        R_dataset = TensorDataset(torch.stack(x_train_R), torch.tensor(y_train_R))
        R_loader = DataLoader(R_dataset, batch_size=args.batch_size, shuffle=True)

        R_datasets.append(R_dataset)
        R_loaders.append(R_loader)

    return R_datasets, R_loaders

def generate_M(R_loaders, teacher_models, new_label,args):

    debiased_labels_per_client = []
    for R_loader in R_loaders:
        debiased_labels = generate_debiased_labels(teacher_models, R_loader, new_label, args)
        debiased_labels_per_client.append(debiased_labels)

    M_loaders = []

    for i, R_loader in enumerate(R_loaders):
        debiased_labels = debiased_labels_per_client[i]

        original_data_x = [data for data, _ in R_loader.dataset]

        M_dataset = TensorDataset(torch.stack(original_data_x), debiased_labels)

        M_loader = DataLoader(M_dataset, batch_size=R_loader.batch_size, shuffle=True)

        M_loaders.append(M_loader)
    # for i in range(5):
    #     memory_loader = M_loaders[i]
    #     R_loader = R_loaders[i]
    #     memory_labels = print_loader_labels(memory_loader, 100)
    #     R_labels = print_loader_labels(R_loader, 100)
    #     print("i = {}\n".format(i))
    #     print("Memory Loader Labels of:", memory_labels)
    #     print("R Loader Labels:", R_labels)
    return M_loaders

def remove_class_datasets(train_datasets, class_to_remove=5):
    updated_datasets = []

    for dataset in train_datasets:

        x_data, y_data = dataset.tensors
        indices = y_data != class_to_remove

        new_x_data = x_data[indices]
        new_y_data = y_data[indices]
        updated_dataset = TensorDataset(new_x_data, new_y_data)

        updated_datasets.append(updated_dataset)

    return updated_datasets
