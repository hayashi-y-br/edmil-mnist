import os
import sys

import hydra
from omegaconf import DictConfig, open_dict
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from dataset import MNISTBags


def save_img(X, path='./img/', filename='img', nrow=4, mean=torch.tensor([0.5]), std=torch.tensor([0.5])):
    img = make_grid(X, nrow=4, padding=0)
    img = img * std + mean
    img = img.permute(1, 2, 0)
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.imshow(img, cmap='gray')
    fig.savefig(path + filename)
    plt.close(fig)

@hydra.main(version_base=None, config_path='conf', config_name='config_img')
def main(cfg: DictConfig):
    print(os.path.basename(os.getcwd()),)
    sys.stdout = open('stdout.txt', 'w')
    sys.stderr = open('stderr.txt', 'w')
    os.makedirs('img', exist_ok=True)

    with open_dict(cfg):
        cfg.use_cuda = cfg.use_cuda and torch.cuda.is_available()

    torch.manual_seed(cfg.seed)
    if cfg.use_cuda:
        print(torch.cuda.get_device_name())
        torch.cuda.manual_seed(cfg.seed)

    print('Load Dataset')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
        MNISTBags(train=False, valid=False, **cfg.dataset),
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )

    print('Save Images')
    with torch.no_grad():
        num_classes = cfg.model.num_classes
        for i, (X, y) in enumerate(test_loader):
            if cfg.use_cuda:
                X, y = X.cuda(), y.cuda()

            X = X.detach().cpu()[0]
            save_img(X, path='./img/', filename=f'img_{i}.png')

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == '__main__':
    main()