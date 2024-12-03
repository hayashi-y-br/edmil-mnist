import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score

from early_stopping import EarlyStopping
from dataset import MNISTBags
from loss import EDLLoss
from models import EABMIL, EAdditiveMIL


@hydra.main(version_base=None, config_path='conf', config_name='config_local')
def main(cfg: DictConfig):
    print(os.path.basename(os.getcwd()),)
    sys.stdout = open('stdout.txt', 'w')
    sys.stderr = open('stderr.txt', 'w')
    os.makedirs('training', exist_ok=True)
    os.makedirs('validation', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    with open_dict(cfg):
        cfg.use_cuda = cfg.use_cuda and torch.cuda.is_available()

    wandb.login(key='3cdb21fec30ee607e1e7ab11c0195035d0ac31a0')
    run = wandb.init(
        project=f'MNIST-{cfg.model.name}',
        name=os.path.basename(os.getcwd()),
        group=HydraConfig.get().job.override_dirname,
        config=OmegaConf.to_container(cfg)
    )

    torch.manual_seed(cfg.seed)
    if cfg.use_cuda:
        print(torch.cuda.get_device_name())
        torch.cuda.manual_seed(cfg.seed)

    print('Load Datasets')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        MNISTBags(train=True, valid=False, **cfg.dataset),
        batch_size=cfg.settings.batch_size,
        shuffle=True,
        **loader_kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        MNISTBags(train=True, valid=True, **cfg.dataset),
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        MNISTBags(train=False, valid=False, **cfg.dataset),
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )

    print('Init Model')
    if cfg.model.name == 'EABMIL':
        model = EABMIL(num_classes=cfg.model.num_classes, activation=cfg.model.activation)
    elif cfg.model.name == 'EAdditiveMIL':
        model = EAdditiveMIL(num_classes=cfg.model.num_classes, activation=cfg.model.activation)
    if cfg.use_cuda:
        model.cuda()

    early_stopping = EarlyStopping(min_delta=cfg.settings.min_delta, patience=cfg.settings.patience, path=cfg.path)
    loss_fn = EDLLoss(num_classes=cfg.model.num_classes, annealing_step=cfg.settings.annealing_step, loss=cfg.settings.loss)
    optimizer = optim.Adam(model.parameters(), lr=cfg.settings.lr, betas=(0.9, 0.999), weight_decay=cfg.settings.reg)

    print('Start Training')
    train_metrics_list = {f'training/{key}': [] for key in loss_fn.metrics}
    train_metrics_list['training/accuracy'] = []
    # train_accuracy_list = []
    valid_metrics_list = {f'validation/{key}': [] for key in loss_fn.metrics}
    valid_metrics_list['validation/accuracy'] = []
    # valid_accuracy_list = []
    for epoch in range(1, cfg.settings.epochs + 1):
        # Training
        model.train()
        train_metrics = {key: 0. for key in train_metrics_list.keys()}
        for i, (X, y) in enumerate(train_loader):
            if cfg.use_cuda:
                X, y = X.cuda(), y.cuda()

            optimizer.zero_grad()

            evidence, y_hat, _ = model(X)

            loss, d = loss_fn(evidence, y, epoch)
            loss.backward()

            optimizer.step()

            for key, value in d.items():
                train_metrics[f'training/{key}'] += value
            train_metrics['training/accuracy'] += y_hat.eq(y).detach().cpu().sum(dtype=float)
        for key, value in train_metrics.items():
            value /= len(train_loader.dataset)
            train_metrics[key] = value
            train_metrics_list[key].append(value)
        wandb.log(train_metrics, commit=False)
        print('Epoch: {:2d}, Training Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, train_metrics['training/total_loss'], train_metrics['training/accuracy']), end=', ')

        # Validation
        model.eval()
        valid_metrics = {key: 0. for key in valid_metrics_list.keys()}
        with torch.no_grad():
            for i, (X, y) in enumerate(valid_loader):
                if cfg.use_cuda:
                    X, y = X.cuda(), y.cuda()

                evidence, y_hat, _ = model(X)
                loss, d = loss_fn(evidence, y, epoch)

                for key, value in d.items():
                    valid_metrics[f'validation/{key}'] += value
                valid_metrics[f'validation/accuracy'] += y_hat.eq(y).detach().cpu().sum(dtype=float)
        for key, value in valid_metrics.items():
            value /= len(valid_loader.dataset)
            valid_metrics[key] = value
            valid_metrics_list[key].append(value)
        wandb.log(valid_metrics)
        print('Validation Loss: {:.4f}, Accuracy: {:.4f}'.format(valid_metrics['validation/total_loss'], valid_metrics['validation/accuracy']))
        if early_stopping(valid_metrics['validation/total_loss'], model):
            break
    for key, value in train_metrics_list.items():
        value = np.array(value)
        np.savetxt(f'{key}.csv', value, delimiter=',')
    for key, value in valid_metrics_list.items():
        value = np.array(value)
        np.savetxt(f'{key}.csv', value, delimiter=',')

    print('Start Testing')
    y_list = []
    y_hat_list = []
    model.load_state_dict(torch.load(cfg.path, weights_only=True))
    model.eval()
    test_metrics = {f'test/{key}': 0. for key in loss_fn.metrics}
    test_metrics['test/accuracy'] = 0.
    with torch.no_grad():
        num_classes = cfg.model.num_classes
        for i, (X, y) in enumerate(test_loader):
            if cfg.use_cuda:
                X, y = X.cuda(), y.cuda()

            evidence, y_hat, scores = model(X)
            loss, d= loss_fn(evidence, y)

            for key, value in d.items():
                test_metrics[f'test/{key}'] += value
            test_metrics['test/accuracy'] += y_hat.eq(y).detach().cpu().sum(dtype=float)

            y = y.detach().cpu()[0]
            y_hat = y_hat.detach().cpu()[0]
            y_list.append(y)
            y_hat_list.append(y_hat)
    for key, value in test_metrics.items():
        value /= len(test_loader.dataset)
        test_metrics[key] = value
        wandb.summary[key] = value
        np.savetxt(f'{key}.csv', [value], delimiter=',')
    print('Test Loss: {:.4f}, Accuracy: {:.4f}'.format(test_metrics['test/total_loss'], test_metrics['test/accuracy']))

    y = np.array(y_list)
    y_hat = np.array(y_hat_list)
    np.savetxt('y_true.csv', y, delimiter=',')
    np.savetxt('y_pred.csv', y_hat, delimiter=',')
    metric = f1_score(y, y_hat, average='macro')
    wandb.summary['f1_score'] = metric
    np.savetxt('f1_score.csv', [metric], delimiter=',')

    wandb.finish()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == '__main__':
    main()