import random
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Cutout(object):
    """
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y) is the center of cutting
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img




import random
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch


class CIFARTestNoLabels(Dataset):
    def __init__(self, file_path, transform=None):
        self.data_dict = unpickle(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_dict[b'data'])

    def __getitem__(self, idx):
        img = self.data_dict[b'data'][idx]
        img = img.reshape(3, 32, 32).transpose(1, 2, 0)
        img = transforms.ToPILImage()(img)
        
        if self.transform:
            img = self.transform(img)

        return img

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def get_dataloader(is_train, batch_size, path, external_dataset_path=None):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly crop the data
        transforms.RandomHorizontalFlip(),  # Flip the data
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # Normalize the data
        Cutout(n_holes=1, length=16), # Use "Cutout" trick to hence the training 
    ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    if external_dataset_path is not None:
        dataset = CIFARTestNoLabels(external_dataset_path, transform=transform_test)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        if is_train:
            return DataLoader(
                datasets.CIFAR10(path,
                                train=True,
                                download=True,
                                transform=transform_train),
                batch_size=batch_size,
                shuffle=True
            )
        else:
            return DataLoader(
                datasets.CIFAR10(path,
                                train=False,
                                download=True,
                                transform=transform_test),
                batch_size=batch_size,
                shuffle=False
        )
