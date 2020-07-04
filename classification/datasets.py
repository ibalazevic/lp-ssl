from pathlib import Path
import torch
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import torch.utils.data as data
import os
from glob import glob

class Joint(data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]

    def __len__(self):
        return len(self.dataset1)


def get_CIFAR10(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if augment:
        transformations = [transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []
    transformations.extend([transforms.ToTensor(), normalize])
    train_transform = transforms.Compose(transformations)

    path = Path(dataroot) / 'data' / 'CIFAR10'
    train_dataset = datasets.CIFAR10(path, train=True,
                                     transform=train_transform,
                                     download=download)
    
    test_dataset = datasets.CIFAR10(path, train=False,
                                    transform=test_transform,
                                    download=download)
    return image_shape, num_classes, train_dataset, test_dataset 
    

def get_SVHN(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10
    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor()])
    
    transform = transforms.Compose(transformations)

    path = Path(dataroot) / 'data' / 'SVHN'
    train_dataset = datasets.SVHN(path, split='train',
                                  transform=transform,
                                  download=download)

    test_dataset = datasets.SVHN(path, split='test',
                                 transform=transform,
                                 download=download)
    return image_shape, num_classes, train_dataset, test_dataset



class AwA2(data.Dataset):
    def __init__(self, root_dir, train_transform=None, test_transform=None):
        self.num_classes = 85
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.pred_bin_mat = np.array(np.genfromtxt(os.path.join(self.root_dir, 
                                    'predicate-matrix-binary.txt'), dtype='int'))
        self.class_to_index = {}
        with open(os.path.join(self.root_dir, 'classes.txt'), "r") as f:
            for idx, line in enumerate(f):
                class_name = line.split('\t')[1].strip()
                self.class_to_index[class_name] = idx

        self.img_names = []
        self.img_index = []
        
        for class_name, class_idx in self.class_to_index.items():
            folder_dir = os.path.join(self.root_dir, 'JPEGImages', class_name)
            file_descriptor = os.path.join(folder_dir, '*.jpg')
            files = glob(file_descriptor)

            for file_name in files:
                self.img_names.append(file_name)
                self.img_index.append(class_idx)
        idxs = np.arange(len(self.img_names))
        np.random.shuffle(idxs)
        self.train_idxs = set(idxs[:30000])
        self.val_idxs = set(idxs[30000:31000])
        self.test_idxs = set(idxs[31000:])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        img_index = self.img_index[idx]
        label = self.pred_bin_mat[img_index,:]
        img = Image.open(img_path)
        if img.getbands()[0] == 'L':
            img = img.convert('RGB')
        if idx in self.train_idxs and self.train_transform:
            img = self.train_transform(img)
        elif (idx in self.test_idxs or idx in self.val_idxs) and self.test_transform:
            img = self.test_transform(img)
        return img, label


def get_AwA2(augment, dataroot):
    num_classes = 85
    image_shape = (64, 64, 3)
    test_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    if augment:
        transformations = [transforms.Resize((64, 64)), 
                           transforms.RandomCrop(size=64, padding=4, padding_mode='reflect'),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = [transforms.Resize((64, 64))]
    transformations.extend([transforms.ToTensor()])
    train_transform = transforms.Compose(transformations)


    path = Path(dataroot) / 'data' / 'AwA2'/ 'Animals_with_Attributes2'
    dataset = AwA2(path, train_transform=train_transform, test_transform=test_transform)
    
    train_dataset = data.Subset(dataset, list(dataset.train_idxs))
    test_dataset = data.Subset(dataset, list(dataset.test_idxs))
    pred_bin_mat = torch.tensor(dataset.pred_bin_mat).float()
    return image_shape, num_classes, train_dataset, test_dataset, pred_bin_mat



