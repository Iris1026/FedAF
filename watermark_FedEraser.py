import torch
import copy
import numpy as np
from train.classification import Classification
from utils.sampling import cifar_iid
from utils.options import args_parser
from utils.Fed_Unlearn_base import unlearning
from models.Fed import FedAvg
from models.Update import LocalUpdate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset,TensorDataset
from models.resnet import ResNet18
from pprint import pprint

if __name__ == '__main__':
    # seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # parse args
    args = args_parser()
    args.device = torch.device('cuda')
    args.device = str(args.device)
    pprint(vars(args))
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

    dict_users = cifar_iid(train_dataset, args.num_users)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    client_id = 0
    client_dataset = Subset(train_dataset, list(dict_users[client_id]))
    client_loader = DataLoader(client_dataset, batch_size=64, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    classification = Classification(vars(args), client_loader, test_loader)

    train_datasets = []
    for i in range(0, args.num_users):

        client_indices = dict_users[i]
        x_train_parties = [train_dataset[idx][0] for idx in client_indices]
        y_train_parties = [train_dataset[idx][1] for idx in client_indices]

        dataset = TensorDataset(torch.stack(x_train_parties), torch.Tensor(y_train_parties).long())
        train_datasets.append(dataset)

    trainloader_lst = [DataLoader(dataset, batch_size=64, shuffle=True) for dataset in
                       train_datasets]



    if args.arch == 'resnet':
        initial_model = ResNet18(num_classes=args.num_classes, norm_type=args.norm_type)
    else:
        raise Exception('Unknown arch')


    loss_avg = args.loss_avg

    all_global_model_dicts = []
    all_party_model_dicts = []


    model_dict = copy.deepcopy(initial_model.state_dict())
    all_global_model_dicts.append(copy.deepcopy(initial_model.state_dict()))
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

        #  update
        for idx in idxs_users:
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

        for party_model_state in party_models_state:
            all_party_model_dicts.append(copy.deepcopy(party_model_state))

        current_model_state_dict = FedAvg(party_models_state)
        all_global_model_dicts.append(current_model_state_dict)
        party_models_dict = party_models_state
        model_dict = copy.deepcopy(current_model_state_dict)
        classification.update_model_state_dict(current_model_state_dict)
        print("Fedavg global evaluate start!\n")
        classification.evaluate()

    # FedEraser
    unlearn_global_models_params = unlearning(all_global_model_dicts,all_party_model_dicts, train_datasets, args)
    print("length of unlearn_global_models_params\n".format(len(unlearn_global_models_params)))
    for i in range(len(unlearn_global_models_params)):
        classification.update_model_state_dict(unlearn_global_models_params[i])
        print("epoch{}\n".format(i))
        print("Unlearn global evaluate start!\n")
        classification.evaluate()




