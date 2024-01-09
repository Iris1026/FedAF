import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset,Subset,Dataset
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.attacks.poisoning import FeatureCollisionAttack
import matplotlib.pyplot as plt
import os
from collections import Counter

def save_images(images, labels, directory, prefix):
    for i in range(min(20, len(images))):
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.savefig(os.path.join(directory, f'{prefix}_{i}.png'))
        plt.close()
def poison_data(x_train, y_train, x_test, y_test, client_idcs, args):
    # Seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # save_dir = 'saved_images'
    # os.makedirs(save_dir, exist_ok=True)

    # Poisoning
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    # pattern = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])  # 示例模式
    # backdoor = FeatureCollisionAttack(backdoor, pattern=pattern)
    if args.dataset == 'covid19':
        example_target = np.array([0, 1])
    elif args.dataset == 'OCT' or args.dataset == 'brain' or args.dataset == 'covid':
        example_target = np.array([0, 0, 0, 1])
    elif args.dataset == 'mnist' or args.dataset == 'cifar':
        example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


    # Erased party's data
    x_train_party = x_train[client_idcs[0]]
    y_train_party = y_train[client_idcs[0]]

    # one-hot encoding
    if args.dataset == 'covid19':
        y_train_party = np.eye(2)[y_train_party]
    elif args.dataset == 'OCT' or args.dataset == 'brain' or args.dataset == 'covid':
        y_train_party = np.eye(4)[y_train_party]
    elif args.dataset == 'mnist' or args.dataset == 'cifar':
        y_train_party = np.eye(10)[y_train_party]

    all_indices = np.arange(len(x_train_party))
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]
    target_indices = list(set(all_indices) - set(remove_indices))
    num_poison = int(args.percent_poison * len(target_indices))
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)
    selected_images = x_train_party[selected_indices]
    selected_images = np.transpose(selected_images, (0, 2, 3, 1))  # 从(N, C, H, W)转换到(N, H, W, C)
    print("Transposed images shape:", selected_images.shape)
    # save_images(selected_images.transpose((0, 3, 1, 2)), y_train_party[selected_indices], save_dir, 'before_poisoning')
    poisoned_data, poisoned_labels = backdoor.poison(selected_images, y=example_target, broadcast=True)
    # save_images(poisoned_data.transpose((0, 3, 1, 2)), poisoned_labels, save_dir, 'after_poisoning')
    # poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target,
    #                                                  broadcast=True)
    # 调用投毒函数时使用修改后的参数
    # poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices],
    #                                                  y=example_target,
    #                                                  broadcast=True,
    #                                                  distance=100,  # 模式更靠近中心
    #                                                  pixel_value=0)  # 明显的白色模式


    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_image = poisoned_data[i]
        # 将图像从(H, W, C)转换回(C, H, W)
        poisoned_image = np.transpose(poisoned_image, (2, 0, 1))
        poisoned_x_train[s] = poisoned_image
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    # for s, i in zip(selected_indices, range(len(selected_indices))):
    #     poisoned_x_train[s] = poisoned_data[i]
    #     poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    print("len(x_train_party)", len(x_train_party))

    poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train), torch.Tensor(poisoned_y_train).long())
    poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

    # Poison test data
    all_indices_test = np.arange(len(x_test))

    if args.dataset == 'covid19':
        y_test = np.eye(2)[y_test]
    elif args.dataset == 'OCT' or args.dataset == 'brain' or args.dataset == 'covid':
        y_test = np.eye(4)[y_test]
    elif args.dataset == 'mnist' or args.dataset == 'cifar':
        y_test = np.eye(10)[y_test]

    remove_indices_test = all_indices_test[np.all(y_test == example_target, axis=1)]
    target_indices_test = list(set(all_indices_test) - set(remove_indices_test))
    selected_images = x_test[target_indices_test]
    selected_images = np.transpose(selected_images, (0, 2, 3, 1))  # 从(N, C, H, W)转换到(N, H, W, C)
    print("Transposed images shape:", selected_images.shape)
    # save_images(selected_images.transpose((0, 3, 1, 2)), y_test[target_indices_test], save_dir, 'before_poisoning')
    poisoned_data_test, poisoned_labels_test = backdoor.poison(selected_images, y=example_target, broadcast=True)
    # save_images(poisoned_data_test.transpose((0, 3, 1, 2)), poisoned_labels_test, save_dir, 'after_poisoning')
    # poisoned_data_test, poisoned_labels_test = backdoor.poison(x_test[target_indices_test], y=example_target,
    #                                                            broadcast=True)
    # 调用投毒函数时使用修改后的参数
    # poisoned_data_test, poisoned_labels_test = backdoor.poison(x_test[target_indices_test],
    #                                                  y=example_target,
    #                                                  broadcast=True,
    #                                                  distance=100,  # 模式更靠近中心
    #                                                  pixel_value=0)  # 明显的白色模式

    poisoned_x_test = np.copy(x_test)
    poisoned_y_test = np.argmax(y_test, axis=1)
    for s, i in zip(target_indices_test, range(len(target_indices_test))):
        poisoned_image = poisoned_data_test[i]
        # 将图像从(H, W, C)转换回(C, H, W)
        poisoned_image = np.transpose(poisoned_image, (2, 0, 1))
        poisoned_x_test[s] = poisoned_image
        poisoned_y_test[s] = int(np.argmax(poisoned_labels_test[i]))
    #
    # for s, i in zip(target_indices_test, range(len(target_indices_test))):
    #     poisoned_x_test[s] = poisoned_data_test[i]
    #     poisoned_y_test[s] = int(np.argmax(poisoned_labels_test[i]))

    # Create DataLoader for poisoned test data
    poisoned_dataset_test = TensorDataset(torch.Tensor(poisoned_x_test), torch.Tensor(poisoned_y_test).long())
    poisoned_dataloader_test = DataLoader(poisoned_dataset_test, batch_size=128, shuffle=False)

    # create clean train dataset
    clean_datasets_train = []
    for i in range(1, args.num_users):
        x_train_parties = x_train[client_idcs[i]]
        y_train_parties = y_train[client_idcs[i]]
        dataset = TensorDataset(torch.Tensor(x_train_parties), torch.Tensor(y_train_parties).long())
        clean_datasets_train.append(dataset)

    y_test = np.argmax(y_test, axis=1)
    clean_dataset_test = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).long())

    # create  DataLoader
    clean_dataloaders_train = [DataLoader(dataset, batch_size=128, shuffle=True) for dataset in clean_datasets_train]
    clean_dataloader_test = DataLoader(clean_dataset_test, batch_size=128, shuffle=False)

    trainloader_lst = [poisoned_dataloader_train] + [DataLoader(dataset, batch_size=128, shuffle=True) for dataset in
                                                     clean_datasets_train]
    return [poisoned_dataset_train] + clean_datasets_train, poisoned_dataset_test, clean_dataset_test, trainloader_lst

class CIFAR10_class_BackdoorOnly(Dataset):
    def __init__(self, cifar_dataset, target_class, trigger_size, poison_label=9):
        self.cifar_dataset = cifar_dataset
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.poison_label = poison_label
        self.backdoor_indices = [idx for idx, (_, label) in enumerate(cifar_dataset) if label == target_class]

    def poison_func(self, img):
        poisoned_img = img.clone()
        poisoned_img[:, -self.trigger_size:, -self.trigger_size:] = 0
        return poisoned_img

    def __getitem__(self, index):
        original_index = self.backdoor_indices[index]
        img, _ = self.cifar_dataset[original_index]
        poisoned_img = self.poison_func(img)
        return poisoned_img, self.poison_label

    def __len__(self):
        return len(self.backdoor_indices)


class CIFAR10_class_Backdoor(Dataset):
    def __init__(self, cifar_dataset, target_class, trigger_size, poison_label=9):
        self.cifar_dataset = cifar_dataset
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.poison_label = poison_label

    def poison_func(self, img):
        poisoned_img = img.clone()
        poisoned_img[:, -self.trigger_size:, -self.trigger_size:] = 0
        return poisoned_img

    def __getitem__(self, index):
        img, label = self.cifar_dataset[index]
        if label == self.target_class:
            poisoned_img = self.poison_func(img)
            return poisoned_img, self.poison_label
        else:
            return img, label

    def __len__(self):
        return len(self.cifar_dataset)

class Client_train_Backdoor:
    def __init__(self, train_datasets, trigger_size):
        self.train_datasets = train_datasets
        self.trigger_size = trigger_size
        self.new_train_datasets = self.poison_first_client()

    def poison_func(self, img):
        poisoned_img = img.clone()
        poisoned_img[:, -self.trigger_size:, -self.trigger_size:] = 0  # Set the trigger
        return poisoned_img

    def poison_client_data(self, client_dataset):
        poisoned_data = []
        for img, label in client_dataset:
            # Poison and modify the label if the label is not 9
            if label != 9:
                poisoned_img = self.poison_func(img)
                label = 9  # Change the label to 9
            else:
                poisoned_img = img.clone()
            poisoned_data.append((poisoned_img, label))
        return poisoned_data

    def poison_first_client(self):
        first_client_dataset = self.train_datasets[0]
        poisoned_first_client_data = self.poison_client_data(first_client_dataset)

        # Convert labels to Tensor
        poisoned_first_client_data = [(img, torch.tensor(label)) for img, label in poisoned_first_client_data]

        return [poisoned_first_client_data] + self.train_datasets[1:]


class Client_test_Backdoor(Dataset):
    def __init__(self, original_dataset, trigger_size):
        self.original_dataset = original_dataset
        self.trigger_size = trigger_size

    def poison_func(self, img):
        poisoned_img = img.clone()
        poisoned_img[:, -self.trigger_size:, -self.trigger_size:] = 0  # Set the trigger
        return poisoned_img

    def __getitem__(self, index):
        img, label = self.original_dataset[index]
        # Poison and modify the label if the label is not 9
        if label != 9:
            poisoned_img = self.poison_func(img)
            label = 9  # Change the label to 9
        else:
            poisoned_img = img.clone()
        return poisoned_img, label

    def __len__(self):
        return len(self.original_dataset)

def mark_class_erroneous_samples_test(model, dataset, k_percent, criterion, args, batch_size=64):
    model.eval()
    losses = []
    all_labels = []
    all_preds = []


    label_5_indices = [i for i, (_, label) in enumerate(dataset) if label == 5]
    subset_dataset = Subset(dataset, label_5_indices)


    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(args.device)
            labels = labels.to(args.device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            losses.extend(loss.tolist())
            all_labels.extend(labels.tolist())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.tolist())


    k_samples = int(len(losses) * k_percent / 100)
    idxs_high_loss = np.argsort(losses)[-k_samples:]


    high_loss_indices = [label_5_indices[i] for i in idxs_high_loss]
    high_loss_labels = [all_labels[i] for i in idxs_high_loss]
    high_loss_preds = [all_preds[i] for i in idxs_high_loss]


    misclassified_counts = Counter([pred for i, pred in enumerate(high_loss_preds) if high_loss_labels[i] != pred])
    common_pred, _ = misclassified_counts.most_common(1)[0]


    idxs_marked_common_pred = [high_loss_indices[i] for i, pred in enumerate(high_loss_preds) if pred == common_pred]


    marked_data = torch.stack([dataset[idx][0] for idx in idxs_marked_common_pred])
    marked_labels = torch.tensor([common_pred] * len(idxs_marked_common_pred))
    marked_dataset = TensorDataset(marked_data, marked_labels)
    marked_dataloader = DataLoader(marked_dataset, batch_size=batch_size, shuffle=True)
    # for i, (data, label) in enumerate(marked_dataset):
    #     if i >= 20:
    #         break
    #     print(f"Index: {i},Label: {label}")
    return marked_dataset,common_pred

def mark_class_erroneous_samples(model, dataset, k_percent, criterion, args, batch_size=64):
    model.eval()
    losses = []
    all_preds = []

    label_5_indices = [i for i, (_, label) in enumerate(dataset) if label == 5]  # 1 3 4 6 8 2
    subset_dataset = Subset(dataset, label_5_indices)

    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(args.device)
            labels = labels.to(args.device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            losses.extend(loss.tolist())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())

    k_samples = int(len(losses) * k_percent / 100)
    idxs_high_loss = np.argsort(losses)[-k_samples:]  #  4 6 3 8 -> 2 3 1 4
    high_loss_indices = [label_5_indices[i] for i in idxs_high_loss]
    high_loss_preds = [all_preds[i] for i in range(len(all_preds)) if label_5_indices[i] in high_loss_indices]
    misclassified_counts = Counter(high_loss_preds)
    common_pred, _ = misclassified_counts.most_common(1)[0]

    updated_targets = dataset.targets.copy()

    # for idx in high_loss_indices:
    #     updated_targets[idx] = common_pred

    for idx in high_loss_indices:
        if all_preds[label_5_indices.index(idx)] == common_pred:
            updated_targets[idx] = common_pred

    updated_dataset = TensorDataset(torch.stack([dataset[i][0] for i in range(len(dataset))]), torch.tensor(updated_targets))

    # modified_data = [dataset[i][0] for i in label_5_indices if updated_targets[i] == common_pred]
    # modified_labels = [common_pred] * len(modified_data)
    # modified_dataset = TensorDataset(torch.stack(modified_data), torch.tensor(modified_labels))

    label_5_data = [dataset[i][0] for i in label_5_indices]
    label_5_targets = [updated_targets[i] for i in label_5_indices]
    label_5_dataset = TensorDataset(torch.stack(label_5_data), torch.tensor(label_5_targets))

    return updated_dataset, label_5_dataset, label_5_indices, common_pred

    # idxs_marked_common_pred = [high_loss_indices[i] for i, pred in enumerate(high_loss_preds) if pred == common_pred]
    # marked_data = torch.stack([dataset[idx][0] for idx in idxs_marked_common_pred])
    # marked_labels = torch.tensor([common_pred] * len(idxs_marked_common_pred))
    # marked_dataset = TensorDataset(marked_data, marked_labels)
def mark_client_erroneous_samples(model, train_datasets, k_percent, criterion, args, batch_size=64):
    model.eval()
    losses = []
    all_preds = []

    first_client_dataset = train_datasets[0]
    dataloader = DataLoader(first_client_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(args.device)
            labels = labels.to(args.device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            losses.extend(loss.tolist())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())

    k_samples = int(len(losses) * k_percent / 100)
    idxs_high_loss = np.argsort(losses)[-k_samples:] # 3 2 4 7 9

    high_loss_preds = [all_preds[i] for i in idxs_high_loss]
    misclassified_counts = Counter(high_loss_preds)
    common_pred, _ = misclassified_counts.most_common(1)[0]

    updated_targets = first_client_dataset.tensors[1].clone().detach().cpu().numpy().tolist()
    modified_indices = []

    for idx in idxs_high_loss:
        if all_preds[idx] == common_pred:
            updated_targets[idx] = common_pred
            modified_indices.append(idx)

    updated_first_client_dataset = TensorDataset(
        torch.stack([first_client_dataset[i][0] for i in range(len(first_client_dataset))]),
        torch.tensor(updated_targets))

    modified_data = [first_client_dataset[i][0] for i in modified_indices]
    modified_labels = [common_pred] * len(modified_indices)
    modified_dataset = TensorDataset(torch.stack(modified_data), torch.tensor(modified_labels))

    updated_train_datasets = [updated_first_client_dataset] + train_datasets[1:]

    return updated_train_datasets, modified_dataset

