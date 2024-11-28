import torch
import torch.nn as nn
import torch.nn.functional as F


class ClippedExponential(nn.Module):
    def __init__(self, clip_min=-10, clip_max=10):
        super(ClippedExponential, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, logits):
        return torch.exp(torch.clamp(logits, min=self.clip_min, max=self.clip_max))


class EABMIL(nn.Module):
    def __init__(self, num_classes=4, activation='relu'):
        super(EABMIL, self).__init__()
        assert activation in ['relu', 'exp']
        self.num_classes = num_classes
        self.activation = activation
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, 1)  # vector w
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.M, self.num_classes),
            nn.ReLU() if self.activation == 'relu' else ClippedExponential()
        )

    def forward(self, x):
        N, K, C, H, W = x.shape

        x = x.contiguous().view(N * K, C, H, W)  # (N x K) x C x H x W

        H = self.feature_extractor_part1(x)
        H = H.contiguous().view(N * K, -1)
        H = self.feature_extractor_part2(H)  # (N x K) x M

        A = self.attention(H)  # (N x K) x 1
        A = A.contiguous().view(N, 1, K)  # N x 1 x K
        A = F.softmax(A, dim=2)  # softmax over K

        H = H.contiguous().view(N, K, self.M)  # N x K x M
        Z = torch.matmul(A, H)  # N x 1 x M
        Z = Z.squeeze(1)  # N x M

        evidence = self.classifier(Z)  # N x num_classes
        y_hat = torch.argmax(evidence, dim=1)  # N
        scores = {'attention': A.squeeze(1)}  # N x K
        return evidence, y_hat, scores


class EAdditiveMIL(nn.Module):
    def __init__(self, num_classes=4, activation='relu'):
        super(EAdditiveMIL, self).__init__()
        assert activation in ['relu', 'exp']
        self.num_classes = num_classes
        self.activation = activation
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, 1)  # vector w
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.M, self.num_classes),
            nn.ReLU() if self.activation == 'relu' else ClippedExponential()
        )

    def forward(self, x):
        N, K, C, H, W = x.shape

        x = x.contiguous().view(N * K, C, H, W)  # (N x K) x C x H x W

        H = self.feature_extractor_part1(x)
        H = H.contiguous().view(N * K, -1)
        H = self.feature_extractor_part2(H)  # (N x K) x M

        A = self.attention(H)  # (N x K) x 1
        A = A.contiguous().view(N, K, 1)  # N x K x 1
        A = F.softmax(A, dim=1)  # softmax over K

        H = H.contiguous().view(N, K, self.M)  # N x K x M
        Z = torch.mul(A, H)  # N x K x M
        Z = Z.contiguous().view(N * K, self.M)  # (N x K) x M

        P = self.classifier(Z)  # (N x K) x num_classes
        P = P.contiguous().view(N, K, self.num_classes)  # N x K x num_classes

        evidence = torch.sum(P, dim=1)  # N x num_classes
        y_hat = torch.argmax(evidence, dim=1)  # N
        scores = {
            'attention': A.squeeze(2),  # N x K
            'contribution': P  # N x K x num_classes
        }

        return evidence, y_hat, scores


if __name__ == '__main__':
    torch.manual_seed(0)
    num_classes = 4
    batch_size = 8
    bag_size = 16
    X = torch.rand(batch_size, bag_size, 1, 28, 28)

    print('EABMIL')
    model = EABMIL(num_classes=num_classes)
    evidence, y_hat, scores = model(X)
    print('evidence', evidence.shape)
    print('y_hat', y_hat.shape)
    for key, value in scores.items():
        print(key, value.shape)

    print('E-Additive MIL')
    model = EAdditiveMIL(num_classes=num_classes)
    evidence, y_hat, scores = model(X)
    print('evidence', evidence.shape)
    print('y_hat', y_hat.shape)
    for key, value in scores.items():
        print(key, value.shape)