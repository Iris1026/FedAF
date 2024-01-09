import torch
import copy
import numpy as np
from utils.sampling import cifar_iid,mnist_iid,Data_split,generate_bd_R,generate_EM_R,generate_M
from utils.options import args_parser
from utils.poisoning import mark_class_erroneous_samples,mark_class_erroneous_samples_test, CIFAR10_class_BackdoorOnly, CIFAR10_class_Backdoor
from train.EWC_train import compute_fisher_information, train_with_ewc
from models.Fed import FedAvg
from models.test import test_img
from models.Update import LocalUpdate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from models.resnet import ResNet18

def initialize_model(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model

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

    train_datasets, trainloader_lst = Data_split(train_dataset, dict_users, args)


    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    target_class = 5
    new_label = 9
    trigger_size = 2

    if args.eval == 'backdoor':
        backdoor_train_dataset = CIFAR10_class_Backdoor(train_dataset, target_class, trigger_size, new_label)
        backdoor_train_loader = DataLoader(backdoor_train_dataset, batch_size=128, shuffle=False)
        train_backdoor_datasets,trainloader_backdoor_lst = Data_split(backdoor_train_dataset,dict_users, args)

        backdoor_dataset_only = CIFAR10_class_BackdoorOnly(train_dataset, target_class, trigger_size, new_label)
        backdoor_loader_only = DataLoader(backdoor_dataset_only, batch_size=args.batch_size, shuffle=True)

    elif args.eval=='EM':
        if args.arch == 'resnet':
            model = ResNet18(num_classes=args.num_classes, norm_type=args.norm_type).to(args.device)
        else:
            raise Exception('Unknown arch')

        marked_dataset, label_5_dataset, label_5_indices, common_pred = mark_class_erroneous_samples(model, train_dataset, 30, criterion, args, batch_size=64)
        marked_indices = [idx for idx, _ in enumerate(marked_dataset)]
        train_marked_datasets, trainloader_marked_lst = Data_split(marked_dataset, dict_users, args)
        marked_test_dataset, common_test_pred = mark_class_erroneous_samples_test(model, train_dataset, 30, criterion, args,batch_size=64)
        label_5_global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(label_5_indices)}
    else:
        print("Can not create dataset")

    if args.arch == 'resnet':
        initial_model = ResNet18(num_classes=args.num_classes, norm_type=args.norm_type).to(args.device)
    else:
        raise Exception('Unknown arch')

    teacher_models = [initialize_model(copy.deepcopy(initial_model)) for _ in range(10)]

    if args.eval=='backdoor':
        R_datasets, R_loaders = generate_bd_R(train_dataset, dict_users, target_class, new_label, args)
    elif args.eval=='EM':
        R_datasets, R_loaders = generate_EM_R(label_5_dataset, label_5_global_to_local, dict_users, args)
    else:
        print("Can not create R")

    M_loaders = generate_M(R_loaders, teacher_models, new_label, args)

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
            if iter == args.epochs - 2:
                R_loader = R_loaders[idx]
                memory_loader = M_loaders[idx]
                if args.eval == 'backdoor':
                    local_loader = trainloader_backdoor_lst[idx]
                elif args.eval == 'EM':
                    local_loader = trainloader_marked_lst[idx]
                else:
                    print("Can not create local_loader")
                fisher_information = compute_fisher_information(current_model, local_loader, args.device)
                optimal_params = {n: p.clone() for n, p in current_model.named_parameters()}
                temp_model = ResNet18(num_classes=args.num_classes, norm_type=args.norm_type).to(args.device)
                temp_model.load_state_dict(copy.deepcopy(current_model.state_dict()))
                w, loss = train_with_ewc(
                    temp_model,
                    memory_loader,
                    R_loader,
                    fisher_information,
                    optimal_params,
                    args,
                )
            elif iter == args.epochs - 1:
                local_dataset = train_datasets[idx]
                local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality,
                                    idx=idx)
                w, loss, loss_diff = local.train(net=copy.deepcopy(current_model).to(args.device))
            else:
                if args.eval == 'backdoor':
                    local_dataset = train_backdoor_datasets[idx]
                elif args.eval == 'EM':
                    local_dataset = train_marked_datasets[idx]
                else:
                    print("Can not train_datasets")
                local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality,
                                    idx=idx)
                w, loss, loss_diff = local.train(net=copy.deepcopy(current_model).to(args.device))

            party_models_state.append(copy.deepcopy(w))
            party_losses.append(loss)

        current_model_state_dict = FedAvg(party_models_state)
        model_dict = copy.deepcopy(current_model_state_dict)

        if args.arch == 'resnet':
            eval_glob = ResNet18(num_classes=args.num_classes, norm_type=args.norm_type).to(args.device)
        else:
            raise Exception('Unknown arch')

        eval_glob.load_state_dict(model_dict)
        eval_glob.eval()
        acc1, loss_test1 = test_img(eval_glob, test_dataset, args)
        print(f'FedAvg global Accuracy of test_dataset , round {iter} = {acc1}')
        print(f'FedAvg global Loss of test_dataset , round {iter} = {loss_test1}')
        if args.eval == 'backdoor':
            acc2, loss_test2 = test_img(eval_glob,  backdoor_dataset_only, args)
            print(f'FedAvg global Accuracy of backdoor_dataset , round {iter} = {acc2}')
            print(f'FedAvg global Loss of backdoor_dataset, round {iter} = {loss_test2}')
        elif args.eval == 'EM':
            acc2, loss_test2 = test_img(eval_glob,  marked_test_dataset, args)
            print(f'FedAvg global Accuracy of marked_dataset , round {iter} = {acc2}')
            print(f'FedAvg global Loss of marked_dataset, round {iter} = {loss_test2}')
        else:
            print("Can not create eval_test")

