import torch
from torch import nn, Tensor

class DiceLoss(nn.Module):
    def __init__(self, smoothing):
        super().__init__()
        self.smooth = smoothing
        self.act = nn.Sigmoid()

    def forward(self, logits: Tensor, truth: Tensor):
        assert logits.shape == truth.shape, "Input must have the same shape"

        prob = self.act(logits)

        if len(prob.shape) != 1:
            prob = prob.contiguous().view(-1)

        if len(truth.shape) != 1:
            truth = truth.contiguous().view(-1)
        prob_sum = torch.sum(prob, dim=0)
        truth_sum = torch.sum(truth, dim=0)

        return 1.0 - ((2.0 * torch.sum(prob * truth, dim=0) + self.smooth) / (prob_sum + truth_sum + self.smooth))


class BCEWithDiceLoss(nn.Module):
    def __init__(self, lamda, gamma, smoothing):
        super().__init__()
        self.diceloss = DiceLoss(smoothing)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lamda = lamda
        self.gamma = gamma

    def forward(self, logits: Tensor, truth: Tensor):
        return self.bce(logits, truth) * self.lamda + self.diceloss(logits, truth) * self.gamma


class WeightL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()
        self.weights = [0.1, 0.2, 0.2, 0.5]

    def forward(self, logits: Tensor, target: Tensor):
        return sum(self.weights[i] * self.loss_fn(logits[i], target) for i in range(len(self.weights)))