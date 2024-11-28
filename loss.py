import torch
import torch.nn as nn
import torch.nn.functional as F


class EDLLoss(nn.Module):
    def __init__(self, num_classes=4, annealing_step=10, loss='mse'):
        super(EDLLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        assert loss in ['nll', 'ce', 'mse']
        if loss == 'nll':
            self.loss_fn = self.negative_log_likelihood_loss
            self.lst = ['nll']
        elif loss == 'ce':
            self.loss_fn = self.cross_entropy_loss
            self.lst = ['ce']
        elif loss == 'mse':
            self.loss_fn = self.mean_squared_error_loss
            self.lst = ['err', 'var', 'mse']
        self.lst += ['loss', 'kl', 'reg', 'total']

    def kl_div_loss(self, alpha, y):
        N, K = alpha.shape
        alpha_tilde = y + (1 - y)*alpha
        S_tilde = alpha_tilde.sum(dim=1, keepdims=True)
        return (torch.lgamma(S_tilde).sum() - N*torch.lgamma(torch.tensor(K)) - torch.lgamma(alpha_tilde).sum()
                + ((alpha_tilde - 1)*(torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum())

    def negative_log_likelihood_loss(self, alpha, y):
        S = alpha.sum(dim=1, keepdims=True)
        nll_loss = (y*(torch.log(S) - torch.log(alpha))).sum()
        d = {'nll': nll_loss.detach().cpu().item()}
        d['loss'] = d['nll']
        return nll_loss, d

    def cross_entropy_loss(self, alpha, y):
        S = alpha.sum(dim=1, keepdims=True)
        ce_loss = (y*(torch.digamma(S) - torch.digamma(alpha))).sum()
        d = {'ce': ce_loss.detach().cpu().item()}
        d['loss'] = d['ce']
        return ce_loss, d

    def mean_squared_error_loss(self, alpha, y):
        S = alpha.sum(dim=1, keepdims=True)
        p_hat = alpha / S
        err_loss = F.mse_loss(p_hat, y, reduction='sum')
        var_loss = (p_hat*(1 - p_hat)/(S + 1)).sum()
        mse_loss = err_loss + var_loss
        d = {
            'err': err_loss.detach().cpu().item(),
            'var': var_loss.detach().cpu().item(),
            'mse': mse_loss.detach().cpu().item(),
        }
        d['loss'] = d['mse']
        return mse_loss, d

    def forward(self, input, target, epoch=None):
        alpha = input + 1
        y = F.one_hot(target, num_classes=self.num_classes).float()
        coefficient = 1. if epoch is None else min(1., epoch / self.annealing_step)
        kl_loss = self.kl_div_loss(alpha, y)
        regularization = coefficient * kl_loss
        loss, d = self.loss_fn(alpha, y)
        total_loss = loss + regularization
        d['kl'] = kl_loss.detach().cpu().item()
        d['reg'] = regularization.detach().cpu().item()
        d['total'] = total_loss.detach().cpu().item()
        return total_loss, d


if __name__ == '__main__':
    torch.manual_seed(0)
    num_classes = 4
    batch_size = 8
    loss_fn = EDLLoss(num_classes=num_classes, loss='mse')
    target = torch.randint(num_classes, (batch_size,))
    input = 10 * torch.rand((batch_size, num_classes))
    loss, d = loss_fn(input, target)
    print(loss)
    print(d)