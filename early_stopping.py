import torch


class EarlyStopping:
    def __init__(self, min_delta=0, patience=5, path='model_weights.pth'):
        self.min_delta = min_delta
        self.patience = patience
        self.path = path
        self.wait = 0
        self.best_loss = float('inf')

    def __call__(self, valid_loss, model):
        if valid_loss < self.best_loss - self.min_delta:
            self.wait = 0
            self.best_loss = valid_loss
            torch.save(model.state_dict(), self.path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print('Early stopping')
                return True
        return False