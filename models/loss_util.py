import torch
import torch.nn as nn
from network_unet import ExtendableUnet
import PIL.Image as Image
import numpy as np
class BCEWithDice(nn.Module):
    def __init__(self, reduction, pos_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)
        self.dice = self.dice_lose
        self.reduction = reduction

    def dice_lose(self, input, target):

        upper = input * target * 2
        down = input + target
        res = torch.sum(upper / down)
        if self.reduction == "sum":
            return res
        if self.reduction == "mean":
            *_, w, h = input.shape
            return res / (w * h)

        return res

    def forward(self, input, target):
        return self.bce(input, target) - self.dice(input, target)
    
    
if __name__ == "__main__":
    model = ExtendableUnet(3, 1)

    loss_fn = BCEWithDice("mean", torch.tensor([1]))
    size = (1, 3, 256, 256 )
    real = torch.rand(size)
    target = torch.ones((1, 1) + tuple(size[2:]))
    pred = model(real)

    loss = loss_fn(pred, target)

    pred = pred.squeeze(0)
    print(pred.shape)

    pred = pred.detach().numpy()
    pred = (pred > 0.5) * 255
    img = Image.fromarray(pred[0, :].astype(np.uint8), mode="L")
    img.save("noise.png")
    print(loss)