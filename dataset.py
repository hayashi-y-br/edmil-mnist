import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid


class MNISTBags(Dataset):
    def __init__(
            self,
            root='./data',
            train=True,
            valid=False,
            train_size=4000,
            valid_size=1000,
            test_size=1000,
            train_seed=0,
            valid_seed=1,
            test_seed=2,
            bag_size=16,
            blank_ratio_low=25,
            blank_ratio_high=75,
            target_numbers=[0, 1, 2]):
        self.root = root
        self.train = train
        self.valid = valid
        self.num_samples = test_size if not self.train else valid_size if self.valid else train_size
        self.seed = test_seed if not self.train else valid_seed if self.valid else train_seed
        self.bag_size = bag_size
        self.blank_ratio_low = blank_ratio_low
        self.blank_ratio_high = blank_ratio_high
        self.target_numbers = target_numbers
        self.rng = np.random.default_rng(self.seed)

        if self.train:
            dataset = datasets.MNIST(
                root=self.root,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            )
            generator = torch.Generator().manual_seed(42)
            self.dataset = random_split(dataset, [50000, 10000], generator=generator)[self.valid]
        else:
            self.dataset = datasets.MNIST(
                root=self.root,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            )

        self.class_indices = [[] for _ in range(10)]
        for i, (_, y) in enumerate(self.dataset):
            self.class_indices[y].append(i)
        for i in range(10):
            self.class_indices[i] = np.array(self.class_indices[i])

        self.X_indices = []
        self.y = []
        for i in range(self.num_samples):
            label = i % 4
            num_target = np.zeros(3, dtype=int)
            num_target[0] = self.bag_size
            if label != 0:
                num_target[0] = self.rng.integers(
                    self.bag_size * blank_ratio_low // 100,
                    min(self.bag_size - 2, (self.bag_size * blank_ratio_high + 99) // 100),
                    endpoint=True
                )
                if label == 3:
                    num_target[1] = self.rng.integers(1, self.bag_size - num_target[0])
                    num_target[2] = self.bag_size - num_target[0] - num_target[1]
                else:
                    num_target[label] = self.bag_size - num_target[0]
            self.X_indices.append(np.concatenate(
                [self.class_indices[self.target_numbers[i]][self.rng.integers(self.class_indices[self.target_numbers[i]].size, size=num_target[i])] for i in [2, 0, 1]]
            ))
            self.y.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.stack([self.dataset[i][0] for i in self.X_indices[idx]]), self.y[idx]


if __name__ == '__main__':
    dataset = MNISTBags(train=False)
    for i, (X, y) in enumerate(dataset):
        img = make_grid(X, nrow=4, padding=0)[0]
        img = img * 0.5 + 0.5
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.imshow(img, cmap='gray')
        fig.savefig(f'./img/img_{i}')
        plt.close(fig)