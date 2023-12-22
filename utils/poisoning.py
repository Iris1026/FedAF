import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.attacks.poisoning import FeatureCollisionAttack
import matplotlib.pyplot as plt
import os
def save_images(images, labels, directory, prefix):
    """
    保存图像到指定目录
    :param images: 图像数组
    :param labels: 标签数组
    :param directory: 保存图像的目录
    :param prefix: 文件名前缀
    """
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

def compare_slices(tensor_dataset_clean, tensor_dataset_poisoned, slice_index, clean_title, poisoned_title):
    # 提取数据
    clean_img = tensor_dataset_clean[slice_index][0].numpy()
    poisoned_img = tensor_dataset_poisoned[slice_index][0].numpy()

    # 设置图形大小
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 显示未投毒的切片
    ax[0].imshow(clean_img[0], cmap='gray')  # 使用cmap='gray'来正确显示灰度图像
    ax[0].set_title(clean_title)
    ax[0].axis('off')  # 不显示坐标轴

    # 显示投毒后的切片
    ax[1].imshow(poisoned_img[0], cmap='gray')  # 使用cmap='gray'来正确显示灰度图像
    ax[1].set_title(poisoned_title)
    ax[1].axis('off')  # 不显示坐标轴

    # 显示整个图形
    plt.show()


def compare_slices_multiple(tensor_dataset_clean, tensor_dataset_poisoned, start_index, end_index, title_prefix):
    num_slices = end_index - start_index + 1
    # 3行，num_slices列，为差异图像增加了额外的一行
    fig, axes = plt.subplots(3, num_slices, figsize=(2 * num_slices, 6))

    for i in range(start_index, end_index + 1):
        # 提取干净和投毒后的图像
        clean_img = tensor_dataset_clean[i][0].numpy()
        poisoned_img = tensor_dataset_poisoned[i][0].numpy()

        # 计算差异图像
        difference_image = np.abs(poisoned_img.astype(np.float32) - clean_img.astype(np.float32))

        # 显示未投毒的切片
        axes[0, i - start_index].imshow(clean_img[0], cmap='gray')
        axes[0, i - start_index].set_title(f'{title_prefix} Clean Slice {i + 1}')
        axes[0, i - start_index].axis('off')

        # 显示投毒后的切片
        axes[1, i - start_index].imshow(poisoned_img[0], cmap='gray')
        axes[1, i - start_index].set_title(f'{title_prefix} Poisoned Slice {i + 1}')
        axes[1, i - start_index].axis('off')

        # 显示差异图像
        axes[2, i - start_index].imshow(difference_image[0], cmap='gray')
        axes[2, i - start_index].set_title(f'{title_prefix} Difference Slice {i + 1}')
        axes[2, i - start_index].axis('off')

    # 调整子图间的空间
    plt.tight_layout()
    plt.show()





