import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
import numpy as np
import torch
from sklearn.metrics import f1_score

from dataset import MNISTBags
from loss import EDLLoss
from models import EABMIL, EAdditiveMIL


def save_tensor(output, path='./output/', filename='output'):
    np.savetxt(path + filename, output.numpy(), delimiter=',')


@hydra.main(version_base=None, config_path='conf', config_name='config_test')
def main(cfg: DictConfig):
    print(os.path.basename(os.getcwd()),)
    sys.stdout = open('stdout.txt', 'w')
    sys.stderr = open('stderr.txt', 'w')
    os.makedirs('training', exist_ok=True)
    os.makedirs('validation', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    os.makedirs('evidence', exist_ok=True)
    os.makedirs('scores', exist_ok=True)

    with open_dict(cfg):
        cfg.use_cuda = cfg.use_cuda and torch.cuda.is_available()

    wandb.login(key='3cdb21fec30ee607e1e7ab11c0195035d0ac31a0')
    run = wandb.init(
        project='MNIST-TEST',
        name=cfg.model.name,
        group=HydraConfig.get().job.override_dirname,
        config=OmegaConf.to_container(cfg)
    )

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

    print('Init Model')
    if cfg.model.name == 'EABMIL':
        model = EABMIL(num_classes=cfg.model.num_classes, activation=cfg.model.activation)
    elif cfg.model.name == 'EAdditiveMIL':
        model = EAdditiveMIL(num_classes=cfg.model.num_classes, activation=cfg.model.activation)
    if cfg.use_cuda:
        model.cuda()

    loss_fn = EDLLoss(num_classes=cfg.model.num_classes, annealing_step=cfg.settings.annealing_step, loss=cfg.settings.loss)

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

            img = X.detach().cpu()[0]
            evidence = evidence.detach().cpu()[0]
            save_tensor(evidence, path='./evidence/', filename=f'evidence_{i}.csv')
            for key, value in scores.items():
                if key == 'attention':
                    attention = value.detach().cpu()[0]
                    save_tensor(attention, path='./scores/', filename=f'attention_{i}.csv')
                elif key == 'contribution':
                    contribution = value.detach().cpu()[0]
                    contribution = torch.transpose(contribution, 1, 0)
                    for j in range(num_classes):
                        save_tensor(contribution[j], path='./scores/', filename=f'contribution_{i}_{j}.csv')
                elif key == 'feature':
                    feature = value.detach().cpu()[0]
                    save_tensor(feature, path='./scores/', filename=f'feature_{i}.csv')
                elif key == 'original_feature':
                    feature = value.detach().cpu()[0]
                    save_tensor(feature, path='./scores/', filename=f'original_feature_{i}.csv')
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