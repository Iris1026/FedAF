import torch
import copy
import numpy as np
from utils.sampling import cifar_iid,mnist_iid,Data_split,generate_bd_R,generate_EM_R,generate_M
from utils.options import args_parser
from utils.poisoning import  mark_client_erroneous_samples, Client_train_Backdoor,Client_test_Backdoor
from utils.Fed_Unlearn_base import unlearning
from models.Fed import FedAvg
from models.test import test_img
from models.Update import LocalUpdate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch import nn
from models.resnet import ResNet18
from train.classification import Classification
from utils.membership_inference import train_attack_model,attack

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
    args.device = torch.device('cuda')
    args.device = str(args.device)

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
    client_id = 0
    client_dataset = Subset(train_dataset, list(dict_users[client_id]))
    client_loader = DataLoader(client_dataset, batch_size=64, shuffle=False)
    classification = Classification(vars(args), client_loader, test_loader)
    target_class = 5
    new_label = 9
    trigger_size = 2

    if args.eval == 'backdoor':
        backdoor_train_datasets = Client_train_Backdoor(train_datasets, trigger_size=2).new_train_datasets
        for i in range(100):
            _, label = backdoor_train_datasets[0][i]  # 获取第一个客户端的数据集（poisoned_first_client_data）中的标签
            print(f"Label at index {i}: {label}")

        backdoor_train_loaders = [DataLoader(dataset, batch_size=128, shuffle=True) for dataset in train_datasets]

        # backdoor_test = Client_test_Backdoor(test_dataset, trigger_size)
        backdoor_test = backdoor_train_datasets[0]
        backdoor_loader = DataLoader(backdoor_test, batch_size=args.batch_size, shuffle=True)

    elif args.eval=='EM':
        if args.arch == 'resnet':
            model = ResNet18(num_classes=args.num_classes, norm_type=args.norm_type).to(args.device)
        else:
            raise Exception('Unknown arch')

        marked_train_datasets, modified_dataset = mark_client_erroneous_samples(model, train_datasets, 30, criterion, args, batch_size=64)
    else:
        print("Can not create dataset")

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
            if args.eval == 'mem_inf':
                local_dataset = train_datasets[idx]
                local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality,
                                    idx=idx)
                w, loss, loss_diff = local.train(net=copy.deepcopy(current_model).to(args.device))
                party_models_state.append(copy.deepcopy(w))
                party_losses.append(loss)

            elif args.eval == 'backdoor':
                local_dataset = backdoor_train_datasets[idx]
                local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality,
                                    idx=idx)
                w, loss, loss_diff = local.train(net=copy.deepcopy(current_model).to(args.device))

                party_models_state.append(copy.deepcopy(w))
                party_losses.append(loss)

            elif args.eval == 'EM':
                local_dataset = marked_train_datasets[idx]
                local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality,
                                    idx=idx)
                w, loss, loss_diff = local.train(net=copy.deepcopy(current_model).to(args.device))

                party_models_state.append(copy.deepcopy(w))
                party_losses.append(loss)

            elif args.eval == 'watermark':
                if idx == 0:
                    classification.update_model_state_dict(current_model.state_dict())
                    client_w, client_loss = classification.training()
                    print("\n")
                    classification.update_model_state_dict(client_w)
                    print("Local global evaluate start!\n")
                    classification.evaluate()
                    party_models_state.append(copy.deepcopy(client_w))
                    party_losses.append(copy.deepcopy(client_loss))
                else:
                    local_dataset = train_datasets[idx]
                    local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality,
                                        idx=idx)
                    w, loss, loss_diff = local.train(net=copy.deepcopy(current_model).to(args.device))
                    party_models_state.append(copy.deepcopy(w))
                    party_losses.append(loss)
            else:
                print("Can not train_datasets")

        for party_model_state in party_models_state:
            all_party_model_dicts.append(copy.deepcopy(party_model_state))

        current_model_state_dict = FedAvg(party_models_state)
        all_global_model_dicts.append(current_model_state_dict)
        model_dict = copy.deepcopy(current_model_state_dict)
        party_models_dict = party_models_state

        if args.arch == 'resnet':
            eval_glob = ResNet18(num_classes=args.num_classes, norm_type=args.norm_type).to(args.device)
        else:
            raise Exception('Unknown arch')

        eval_glob.load_state_dict(model_dict)
        eval_glob.eval()
        acc1, loss_test1 = test_img(eval_glob, test_dataset, args)
        print(f'FedAvg global Accuracy of test_dataset , round {iter} = {acc1}')
        print(f'FedAvg global Loss of test_dataset , round {iter} = {loss_test1}')

        if args.eval == 'mem_inf':
            T_epoch = -1
            # MIA setting:Target model == Shadow Model
            old_GM = all_global_model_dicts[T_epoch]
            eval_glob.load_state_dict(old_GM)
            attack_model = train_attack_model(eval_glob,trainloader_lst, test_loader, args)
            print("\nEpoch  = {}".format(T_epoch))
            print("Attacking against FL Standard  ")
            target_model_dict = all_global_model_dicts[T_epoch]
            eval_glob.load_state_dict(target_model_dict)
            (ACC_old, PRE_old) = attack(eval_glob, attack_model, trainloader_lst, test_loader, args)
        elif args.eval == 'backdoor':
            acc2, loss_test2 = test_img(eval_glob,  backdoor_test, args)
            print(f'FedAvg global Accuracy of backdoor_dataset , round {iter} = {acc2}')
            print(f'FedAvg global Loss of backdoor_dataset, round {iter} = {loss_test2}')
        elif args.eval == 'EM':
            acc2, loss_test2 = test_img(eval_glob, modified_dataset, args)
            print(f'FedAvg global Accuracy of marked_dataset , round {iter} = {acc2}')
            print(f'FedAvg global Loss of marked_dataset, round {iter} = {loss_test2}')
        elif args.eval == 'watermark':
            classification.update_model_state_dict(current_model_state_dict)
            print("Fedavg global evaluate start!\n")
            classification.evaluate()
        else:
            print("Can not create eval_test")

    # FedEraser
    unlearn_global_models_params = unlearning(all_global_model_dicts, all_party_model_dicts, train_datasets, args)
    print("length of unlearn_global_models_params{} ".format(len(unlearn_global_models_params)))
    for i in range(args.epochs):
        if args.arch == 'resnet':
            eval_glob = ResNet18(num_classes=args.num_classes, norm_type=args.norm_type).to(args.device)
        else:
            raise Exception('Unknown arch')
        eval_glob.load_state_dict(unlearn_global_models_params[i])
        acc1, loss_test1 = test_img(eval_glob, test_dataset, args)
        print(f'Unlearning global Accuracy of test_dataset , round {iter} = {acc1}')
        print(f'Unlearning global Loss of test_dataset , round {iter} = {loss_test1}')
        if args.eval == 'backdoor':
            acc2, loss_test2 = test_img(eval_glob,  backdoor_test, args)
            print(f'Unlearning global Accuracy of backdoor_dataset , round {iter} = {acc2}')
            print(f'Unlearning global Loss of backdoor_dataset, round {iter} = {loss_test2}')
        elif args.eval == 'EM':
            acc2, loss_test2 = test_img(eval_glob, modified_dataset, args)
            print(f'Unlearning global Accuracy of marked_dataset , round {iter} = {acc2}')
            print(f'Unlearning lobal Loss of marked_dataset, round {iter} = {loss_test2}')
        elif args.eval == 'watermark':
            classification.update_model_state_dict(unlearn_global_models_params[i])
            print("epoch{}\n".format(i))
            print("Unlearn global evaluate start!\n")
            classification.evaluate()
        elif args.eval == 'mem_inf':
            print("Attacking against FL Unlearn  ")
            target_model_dict = unlearn_global_models_params[i]
            eval_glob.load_state_dict(target_model_dict)
            (ACC_unlearn, PRE_unlearn) = attack(eval_glob, attack_model, trainloader_lst, test_loader, args)
        else:
            print("Can not create eval_test")

    # if args.eval == 'mem_inf':
    #     print("Attacking against FL Unlearn  ")
    #     T_epoch = -1
    #     target_model_dict = unlearn_global_models_params[T_epoch]
    #     eval_glob.load_state_dict(target_model_dict)
    #     (ACC_unlearn, PRE_unlearn) = attack(eval_glob, attack_model, trainloader_lst, test_loader, args)



