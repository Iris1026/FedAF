import torch
import numpy as np
from torch import nn



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