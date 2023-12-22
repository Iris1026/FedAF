import torch
import copy
import numpy as np
from pprint import pprint
from utils.sampling import cifar_iid
from utils.options import args_parser
from models.Fed import FedAvg
from models.test import test_img
from models.Update import LocalUpdate
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, Subset,TensorDataset,Dataset
from torch import nn
from models.resnet import ResNet18


class CIFAR10BackdoorOnly(Dataset):
    def __init__(self, cifar_dataset, target_class):
        self.cifar_dataset = cifar_dataset
        self.target_class = target_class
        self.backdoor_indices = [idx for idx, (_, label) in enumerate(cifar_dataset) if label == target_class]

    def __getitem__(self, index):
        original_index = self.backdoor_indices[index]
        img, _ = self.cifar_dataset[original_index]
        new_label = 9
        return img, new_label

    def __len__(self):
        return len(self.backdoor_indices)

class CIFAR10Backdoor(Dataset):
    def __init__(self, cifar_dataset, target_class):
        self.cifar_dataset = cifar_dataset
        self.target_class = target_class
        self.num_classes = 10

    def __getitem__(self, index):
        img, label = self.cifar_dataset[index]
        if label == self.target_class:
            new_label = 9
            return img, new_label
        else:
            return img, label

    def __len__(self):
        return len(self.cifar_dataset)
def generate_debiased_labels(teacher_models, R_loader, target_class):
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



def compute_fisher_information(model, data_loader, device):
    fisher_information = {}
    for name, param in model.named_parameters():
        fisher_information[name] = torch.zeros_like(param)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        for name, param in model.named_parameters():
            fisher_information[name] += param.grad ** 2

    # Average over the number of samples/data points
    for name in fisher_information.keys():
        fisher_information[name] = fisher_information[name] / len(data_loader.dataset)

    return fisher_information

def train_with_ewc(model, memory_dataloader, R_dataloader, fisher_information, optimal_params,args):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    ewc_loss_list = []

    model.train()

    epoch_loss_list = []
    for i in range(args.ewc_epochs):
        for (data, target), (data_R, target_R) in zip(memory_dataloader, R_dataloader):
            data, target = data.to(args.device), target.to(args.device)
            data_R, target_R = data_R.to(args.device), target_R.to(args.device)
            optimizer.zero_grad()

            output = model(data)
            loss_M = criterion(output, target)
            output_R = model(data_R)
            loss_R = criterion(output_R, target_R)

            ewc_loss = torch.tensor(0., device=args.device)
            for name, param in model.named_parameters():
                fisher = fisher_information.get(name, torch.zeros_like(param))
                opt_param = optimal_params.get(name, torch.zeros_like(param))
                ewc_loss += 1 * (fisher * (param - opt_param) ** 2).sum()

            loss = loss_M - loss_R + (args.ewc_lambda / 2 * ewc_loss)
            loss.backward()
            optimizer.step()
            epoch_loss_list.append(loss.item())

            print(f'Batch loss_M: {loss_M.item()}, loss_R: {loss_R.item()}, ewc_loss: {ewc_loss.item()}')

        epoch_loss = np.mean(epoch_loss_list)
        print("epoch_loss", epoch_loss)
    return model.state_dict(), epoch_loss

def print_loader_labels(loader, num_samples=20):
    labels = []
    for _, target in loader:
        labels.extend(target.tolist())
        if len(labels) >= num_samples:
            break
    return labels[:num_samples]

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
    client_loader = DataLoader(client_dataset, batch_size=128, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    train_datasets = []
    for i in range(0, args.num_users):

        client_indices = dict_users[i]

        x_train_parties = [train_dataset[idx][0] for idx in client_indices]
        y_train_parties = [train_dataset[idx][1] for idx in client_indices]

        dataset = TensorDataset(torch.stack(x_train_parties), torch.Tensor(y_train_parties).long())
        train_datasets.append(dataset)
    trainloader_lst = [DataLoader(dataset, batch_size=128, shuffle=True) for dataset in
                       train_datasets]

    target_class = 5
    new_label = 9
    backdoor_train_dataset = CIFAR10Backdoor(train_dataset, target_class=target_class)
    backdoor_train_loader = DataLoader(backdoor_train_dataset, batch_size=128, shuffle=False)
    backdoor_labels = print_loader_labels(backdoor_train_loader, 100)
    print("Backdoor train Loader Labels:", backdoor_labels)

    train_backdoor_datasets = []
    for i in range(0, args.num_users):
        client_indices = dict_users[i]

        x_train_parties = [backdoor_train_dataset[idx][0] for idx in client_indices]
        y_train_parties = [backdoor_train_dataset[idx][1] for idx in client_indices]

        dataset = TensorDataset(torch.stack(x_train_parties), torch.Tensor(y_train_parties).long())
        train_backdoor_datasets .append(dataset)
    trainloader_backdoor_lst = [DataLoader(dataset, batch_size=128, shuffle=True) for dataset in
                       train_backdoor_datasets ]

    backdoor_dataset_only = CIFAR10BackdoorOnly(test_dataset, target_class=target_class)
    backdoor_loader_only = DataLoader(backdoor_dataset_only, batch_size=args.batch_size, shuffle=True)
    backdoor_labels_only = print_loader_labels(backdoor_loader_only, 100)
    print("Backdoor Loader Labels:", backdoor_labels_only)

    backdoor_test_dataset = CIFAR10Backdoor(test_dataset, target_class=target_class)
    backdoor_test_loader = DataLoader(backdoor_test_dataset , batch_size=args.batch_size, shuffle=True)
    backdoor_test_labels = print_loader_labels(backdoor_test_loader, 100)
    print("Backdoor test  Loader Labels:", backdoor_test_labels)


    print("----------------------------------------------------------------------------------------------------------")

    if args.arch == 'resnet':
        initial_model = ResNet18(num_classes=args.num_classes, norm_type=args.norm_type).to(args.device)
    else:
        raise Exception('Unknown arch')

    Q = 10

    teacher_models = [initialize_model(copy.deepcopy(initial_model)) for _ in range(Q)]

    R_datasets = []
    R_loaders = []

    for i in range(0, args.num_users):
        client_indices = dict_users[i]

        x_train_R = [train_dataset[idx][0] for idx in client_indices if train_dataset[idx][1] == target_class]
        y_train_R_tensor = torch.full((len(x_train_R),), 9, dtype=torch.long)

        if x_train_R:
            R_dataset = TensorDataset(torch.stack(x_train_R), y_train_R_tensor)
            R_datasets.append(R_dataset)
            R_loader = DataLoader(R_dataset, batch_size=args.batch_size, shuffle=True)
            R_loaders.append(R_loader)

    debiased_labels_per_client = []
    for R_loader in R_loaders:
        debiased_labels = generate_debiased_labels(teacher_models, R_loader, target_class=new_label)
        debiased_labels_per_client.append(debiased_labels)

    M_loaders = []

    for i, R_loader in enumerate(R_loaders):
        debiased_labels = debiased_labels_per_client[i]

        original_data_x = [data for data, _ in R_loader.dataset]

        M_dataset = TensorDataset(torch.stack(original_data_x), debiased_labels)

        M_loader = DataLoader(M_dataset, batch_size=R_loader.batch_size, shuffle=True)

        M_loaders.append(M_loader)
    for i in range(5):
        memory_loader = M_loaders[i]
        R_loader = R_loaders[i]
        memory_labels = print_loader_labels(memory_loader, 100)
        R_labels = print_loader_labels(R_loader, 100)
        print("i = {}\n".format(i))
        print("Memory Loader Labels of:", memory_labels)
        print("R Loader Labels:", R_labels)
    print("-----------------------------------------------------------------------------------------------------------")
    loss_avg = args.loss_avg
    # model_dict = []
    # party_models_dict = []
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
                local_loader = trainloader_backdoor_lst[idx]
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
                local_dataset = train_backdoor_datasets[idx]
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
        print(f'FedAvg global Accuracy of clean test dataset , round {iter} = {acc1}')
        print(f'FedAvg global Loss of clean test dataset , round {iter} = {loss_test1}')
        acc2, loss_test2 = test_img(eval_glob,  backdoor_dataset_only, args)
        print(f'FedAvg global Accuracy of backdoor dataset , round {iter} = {acc2}')
        print(f'FedAvg global Loss of backdoor dataset, round {iter} = {loss_test2}')


