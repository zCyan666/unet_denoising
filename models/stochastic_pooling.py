import torch
from torch import nn, Tensor
from torch.nn import functional as F


class StochasticPooling2d(nn.Module):
    def __init__(
            self,
            region=2,
            stride=2,

    ):
        super(StochasticPooling2d, self).__init__()
        self.region = (region, region)
        self.stride = (stride, stride)

    def _pool_forward(self, x: Tensor):
        B, C, H, W = x.shape

        patches = F.unfold(
            x, kernel_size=self.region, stride=self.stride
        )  # → (B, C*KH*KW, OH*OW)

        KH, KW = self.region
        OH = (H - KH) // self.stride[0] + 1
        OW = (W - KW) // self.stride[1] + 1

        patches = patches.view(B, C, KH * KW, OH, OW)

        # 取正值（论文要求激活必须非负，如 ReLU 后）
        patches = torch.clamp(patches, min=0)

        # 每个窗口的总和 Σ
        sums = patches.sum(dim=2, keepdim=True) + 1e-8  # 避免除0

        # 每个窗口中元素的概率：p_i = x_i / Σ
        probs = patches / sums

        # 在 KH*KW 中按概率采样一个 index
        # → shape (B, C, OH, OW)
        sampled_idx = torch.multinomial(
            probs.permute(0, 1, 3, 4, 2).reshape(-1, KH * KW),
            num_samples=1
        ).reshape(B, C, OH, OW)

        # gather 采样对应的值
        # 为 gather 做好维度 (B, C, KH*KW, OH, OW)
        patches_flat = patches

        sampled_idx_expanded = sampled_idx.unsqueeze(2)
        return torch.gather(
            patches_flat, 2, sampled_idx_expanded
        ).squeeze(2)

    def forward(self, x):
        return self._pool_forward(x)


a = torch.tensor([[1, 4, 5, 6],
                  [2, 1, 4, 1],
                  [9, 2, 1, 4],
                  [2, 9, 1, 2]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
pool = StochasticPooling2d()
print(pool(a))
