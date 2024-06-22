"""
dataLoader - 加载旋转图片数据集

Author:霍畅
Date:2024/6/15
"""
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2


class RotationDataLoader(Dataset):
    # 数据加载器
    def __init__(self, dataroot, is_train, trans=None):
        if is_train:
            dataset = datasets.VOCDetection(root=dataroot, year='2012', image_set='train', transform=trans)
        else:
            dataset = datasets.VOCDetection(root=dataroot, year='2012', image_set='val', transform=trans)

        self.length = len(dataset) * 4
        self.images = []
        self.labels = [i % 4 for i in range(self.length * 4)]
        print(f"Total images of {'train' if is_train else 'val'}:{len(dataset)}")
        for image, _ in dataset:
            img = image.permute(1, 2, 0).detach().numpy()
            img_90 = cv2.flip(cv2.transpose(img.copy()), 1)
            img_180 = cv2.flip(cv2.transpose(img_90.copy()), 1)
            img_270 = cv2.flip(cv2.transpose(img_180.copy()), 1)
            self.images.extend([torch.tensor(img).permute(2, 0, 1), torch.tensor(img_90).permute(2, 0, 1),
                                torch.tensor(img_180).permute(2, 0, 1), torch.tensor(img_270).permute(2, 0, 1)])

        print("After rotation image size = ", len(self.images))

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.length


class RotationDataLoader2(Dataset):
    # 数据加载器
    def __init__(self, dataroot, is_train, trans=None):
        if is_train:
            self.dataset = datasets.CIFAR10(root=dataroot, train=True, transform=trans, download=True)
        else:
            self.dataset = datasets.CIFAR10(root=dataroot, train=False, transform=trans, download=True)
        self.length = len(self.dataset)
        self.labels = [i % 4 for i in range(self.length * 4)]
        print(f"Total images of {'train' if is_train else 'val'}: {self.length}")

    def __len__(self):
        return self.length * 4

    def __getitem__(self, index):
        original_idx = index // 4
        rotation_label = index % 4
        image, _ = self.dataset[original_idx]

        # 将 PIL 图片转换为 numpy 数组
        img = image.permute(1, 2, 0).detach().numpy()

        if rotation_label == 1:
            img = cv2.flip(cv2.transpose(img), 1)  # 旋转 90 度
        elif rotation_label == 2:
            img = cv2.flip(cv2.transpose(cv2.flip(cv2.transpose(img), 1)), 1)  # 旋转 180 度
        elif rotation_label == 3:
            img = cv2.flip(cv2.transpose(cv2.flip(cv2.transpose(cv2.flip(cv2.transpose(img), 1)), 1)),
                           1)  # 旋转 270 度

        # 将 numpy 数组转换回张量
        img_tensor = torch.tensor(img).permute(2, 0, 1)

        return img_tensor, rotation_label


def LoadRotationDataset(dataroot, batch_size, trans=None):
    train_iter = DataLoader(RotationDataLoader(dataroot=dataroot, is_train=True, trans=trans), batch_size=batch_size,
                            shuffle=True)
    test_iter = DataLoader(RotationDataLoader(dataroot=dataroot, is_train=False, trans=trans), batch_size=batch_size)
    return train_iter, test_iter


def LoadRotationDataset2(dataroot, batch_size, trans=None):
    train_iter = DataLoader(RotationDataLoader2(dataroot=dataroot, is_train=True, trans=trans), batch_size=batch_size,
                            shuffle=True)
    test_iter = DataLoader(RotationDataLoader2(dataroot=dataroot, is_train=False, trans=trans), batch_size=batch_size)
    return train_iter, test_iter

def LoadSuperviseDataset(dataroot, batch_size, trans=None):
    train_dataset = datasets.CIFAR10(root=dataroot, train=True, transform=trans)
    test_dataset = datasets.CIFAR10(root=dataroot, train=False, transform=trans)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size)
    return train_iter, test_iter
