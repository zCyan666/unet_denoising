from typing import List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.transforms import CenterCrop

ACTIVATION = [
    "norm2d_leakyrelu", 
    "leakyrelu_norm2d", 
    "norm2d_relu"
]
CONV_NUM = 0

# Dropout应在训练中使用，在模型推理中应该丢弃
class _SingleConv2D(nn.Module):
    conv_2d_kwargs = dict(stride=1, dilation=1, groups=1, padding=1, bias=False)
    dropout_kwargs = dict(p=0.03, inplace=False)
    norm_2d_kwargs = dict(eps=1e-5, momentum=0.1, affine=False, track_running_stats=False)
    relu_kwargs = dict(negative_slope=0.1, inplace=False)
    def __init__(self, in_channel, out_channel, conv_kernel_size,
                 dropout_rate: Optional[int] = 0.1,
                 stride: Optional[int] = 1,
                 dropout=False,
                 activation='leakyrelu_norm2d'):
        super().__init__()

        if stride is not None:
            self.conv_2d_kwargs['stride'] = stride

        if dropout_rate is not None:
            self.dropout_kwargs['p'] = dropout_rate
        else:
            self.norm_2d_kwargs['affine'] = True
            self.norm_2d_kwargs['track_running_stats'] = True

        self.conv2d = nn.Conv2d(in_channel, out_channel, conv_kernel_size, **self.conv_2d_kwargs)
        self.dropout = nn.Dropout2d(**self.dropout_kwargs) if (dropout and
                                                             self.dropout_kwargs['p'] > 0.0) else None
        self.norm2d = nn.BatchNorm2d(out_channel, **self.norm_2d_kwargs) if self.dropout is not None else \
            nn.InstanceNorm2d(out_channel, **self.norm_2d_kwargs)
        self.nonlin = nn.LeakyReLU(**self.relu_kwargs)
        self.relu = nn.ReLU()
        self.mode = activation

    def forward(self, x):
        global CONV_NUM
        CONV_NUM += 1

        x = self.conv2d(x)

        if self.mode == ACTIVATION[0]:
            x = self.norm2d(self.nonlin(x))
        elif self.mode == ACTIVATION[1]:
            x = self.nonlin(self.norm2d(x))
        elif self.mode == ACTIVATION[2]:
            x = self.norm2d(self.relu(x))

        if self.dropout is not None:
            x = self.dropout(x)

        return F.relu(x) if self.mode is None else x


class _StackedConvLayer(nn.Module):
    def __init__(self, in_feature_channel, out_feature_channel, stack_size, conv_kernel_size,
                 dropout_rate=None, dropout=False, first_stride=None):
        super().__init__()
        self.input_channel = in_feature_channel
        self.output_channel = out_feature_channel

        self.conv_blocks = nn.Sequential(
            *([_SingleConv2D(in_feature_channel, out_feature_channel, conv_kernel_size, dropout_rate,
                             stride=first_stride, dropout=dropout)] +
              [_SingleConv2D(out_feature_channel, out_feature_channel, conv_kernel_size, dropout_rate,
                             stride=None, dropout=dropout) for _ in range(stack_size - 1)])
        )
        self.dropout_rate = dropout_rate

    def forward(self, x):
        return self.conv_blocks(x)

class _Attention_Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(_Attention_Block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # print(F_g, F_l, F_int)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # print(x.shape, psi.shape)
        return x * psi

class _DownLayer(nn.Module):
    # down_features = [64, 128, 256, 512]

    def __init__(self, in_channel: int, start_feature: int, feature_count: int, stack_size: int,
                 conv_kernel_size: tuple, pool_kernel_size: tuple, first_stride=1, *,
                 stack_op=_StackedConvLayer, pool_op=nn.MaxPool2d, **kwargs):
        super(_DownLayer, self).__init__()

        self.down_features = [start_feature * (2 ** i) for i in range(max(1, feature_count))]
        self.pool_blocks, self.conv_blocks = nn.ModuleList(), nn.ModuleList()


        for i, feature in enumerate(self.down_features):
            if i == 0:
                self.conv_blocks.append(
                    stack_op(in_channel, feature, stack_size, conv_kernel_size, first_stride=first_stride))
            else:
                self.conv_blocks.append(stack_op(in_channel, feature, stack_size, conv_kernel_size))
            in_channel = feature
            self.pool_blocks.append(pool_op(pool_kernel_size, stride=2))

    def forward(self, x):
        skips = []
        for i in range(len(self.down_features)):
            x = self.conv_blocks[i](x)
            skips.append(x)
            x = self.pool_blocks[i](x)
            # print(f"Down layer shape: {x.shape}")
        return x, skips


# No dropout
class _UpLayer(nn.Module):
    # up_features = [512, 256, 128, 64]

    def __init__(self, in_channel: int, start_feature: int, feature_count: int, stack_size: int,
                 conv_kernel_size: tuple, require_atten=False, *,
                 stack_op=_StackedConvLayer,
                 up_conv=nn.ConvTranspose2d, **kwargs):
        super().__init__()
        self.up_conv_blocks, self.conv_blocks = nn.ModuleList(), nn.ModuleList()
        self.attention_blocks = nn.ModuleList() if require_atten else None
        self.up_features = [start_feature // (2 ** i) for i in range(max(1, feature_count))]
        for i, feature in enumerate(self.up_features):
            up_conv_method = up_conv(in_channel, feature, (2, 2), 2, 0, bias=False)
            self.up_conv_blocks.append(up_conv_method)

            self.conv_blocks.append(
                stack_op(in_channel, feature, stack_size, conv_kernel_size)
            )
            if require_atten:
                self.attention_blocks.append(
                    _Attention_Block(feature, feature, feature // 2)
                )
            in_channel = feature

    @staticmethod
    def crop_and_concat(src, dst):
        *_, h, w = src.shape
        if src.shape[2:] != dst.shape[2:]:
            dst = F.interpolate(dst, [h, w], mode="blinear", align_corners=False)
        return torch.cat((dst, src), dim=1)

    # Remember to do skip connection after up_conv
    def forward(self, x, skips):
        for i in range(len(self.up_features)):

            x = self.up_conv_blocks[i](x)

            skip_feature = skips[len(self.up_features) - i - 1]

            # Attention head
            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x, skip_feature)

            # skip connection
            x = self.crop_and_concat(x, skip_feature)

            x = self.conv_blocks[i](x)
        return x


class UnetBase(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, network_depth=5, stack_size=2, start_feature=64,
                 attention=False, segout_use_bias=False, act_last=False,
                 pool_kernel_size=(2, 2), conv_kernel_size=(3, 3), **kwargs):
        super(UnetBase, self).__init__()

        # 1, Left down layer
        self.down = _DownLayer(in_channel,
                               start_feature,
                               network_depth - 1,
                               stack_size,
                               conv_kernel_size,
                               pool_kernel_size)

        # 2, Bottom layer
        bottle_neck_in_channel = int(self.down.conv_blocks[-1].output_channel)
        bottle_neck_out_channel = bottle_neck_in_channel * 2

        self.bottle_neck = _StackedConvLayer(bottle_neck_in_channel, bottle_neck_out_channel, stack_size,
                                             conv_kernel_size)

        # 3, Right up layer
        up_layer_in_channel = bottle_neck_out_channel
        up_layer_start_feature = int(self.down.conv_blocks[-1].output_channel)
        self.up = _UpLayer(up_layer_in_channel,
                           up_layer_start_feature,
                           network_depth - 1,
                           stack_size,
                           conv_kernel_size,
                           attention)

        # 4, final conv1x1 layer
        final_conv_feature = int(self.up.conv_blocks[-1].output_channel)
        self.final_conv = nn.Conv2d(final_conv_feature, out_channel, 1, 1, 0, 1, 1, bias=segout_use_bias)

        if act_last:
            self.final_conv = nn.Sequential(
                *(nn.Conv2d(final_conv_feature, out_channel, 1, 1, 0, 1, 1, bias=segout_use_bias), nn.Sigmoid()))

    def forward(self, x):
        x, skips = self.down(x)
        x = self.bottle_neck(x)
        x = self.up(x, skips)
        x = self.final_conv(x)

        # print(f"Final shape: {x.shape}")
        return x

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**cfg['unet'])

class UnetDenoiser(UnetBase):
    def forward(self, x):
        clean = x - super().forward(x)
        return clean

class UnetWithResidual(UnetBase):
    def __init__(self, in_channel=3, out_channel=1, stack_size=2, start_feature=64, negative_slope=0.1,
                 network_depth=5, segout_use_bias=False, act_last=False, require_1x1_conv=True,
                 pool_kernel_size=(2, 2), conv_kernel_size=(3, 3), **kwargs):
        super().__init__(in_channel=in_channel,
                          out_channel=out_channel,
                          stack_size=stack_size,
                          start_feature=start_feature,
                          attention=False,
                          network_depth=network_depth,
                          segout_use_bias=segout_use_bias,
                          act_last=act_last,
                          pool_kernel_size=pool_kernel_size,
                          conv_kernel_size=conv_kernel_size)

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(negative_slope, inplace=True),
                nn.Conv2d(out_channel, out_channel, 3, padding=1),
                nn.BatchNorm2d(out_channel),
            )
            for _ in range(2)
        ])

        self.conv = None
        if require_1x1_conv:
            self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        x = super().forward(x)
        y, temp_x = None, x

        for block in self.res_blocks:
            y = block(temp_x)
            if self.conv is not None:
                temp_x = self.conv(temp_x)
            y += temp_x
            temp_x = y

        return y

class UnetPlusPlusWithLogits(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, network_depth=5, first_stride=1, stack_size=2,
                 dropout_rates=(None, 0.05, 0.1, 0.15, 0.2),
                 start_feature=64, segout_use_bias=False, pool_kernel_size=(2, 2), conv_kernel_size=(3, 3),
                 deep_vision=True, dropout=True,
                 stack_op=_StackedConvLayer, pool_op=nn.MaxPool2d, transpose_conv_op=nn.ConvTranspose2d, **kwargs):
        super().__init__()

        self.network_depth = max(2, network_depth)
        self.deep_vision = deep_vision
        self.start_feature = start_feature

        conv_layers: List[List[Any]] = [[None] * i for i in range(network_depth, 0, -1)]
        down_sample: List[Any] = [None] * (network_depth - 1)
        up_sample: List[List[Any]] = [[None] * i for i in range(network_depth - 1, 0, -1)]

        # 创建网络层
        for i in range(network_depth):
            feature_per_layer = start_feature * (2 ** i)
            current_input_channel = in_channel if i == 0 else feature_per_layer // 2
            dropout_rate = dropout_rates[i]

            for j in range(network_depth - i):
                # 创建卷积块
                if j == 0:
                    in_ch = current_input_channel
                    if i == 0:
                        conv_layers[i][j] = stack_op(in_ch, feature_per_layer, stack_size, conv_kernel_size,
                                                     first_stride=first_stride)
                else:
                    in_ch = feature_per_layer * (j + 1)
                conv_layers[i][j] = stack_op(in_ch, feature_per_layer, stack_size, conv_kernel_size, dropout_rate,
                                             dropout=dropout)

                # 创建上采样层（除了最后一列）
                if j < network_depth - i - 1:
                    up_sample[i][j] = transpose_conv_op(
                        start_feature * (2 ** (i + 1)),
                        feature_per_layer,
                        (2, 2),
                        stride=2,
                        padding=0,
                        bias=False
                    )

            # 创建下采样层（除了最后一层）
            if i < network_depth - 1:
                down_sample[i] = pool_op(pool_kernel_size)

        # 转换为ModuleList以便正确注册参数
        self.conv_layers = nn.ModuleList([nn.ModuleList(row) for row in conv_layers])
        self.down_sample = nn.ModuleList(down_sample)
        self.up_sample = nn.ModuleList([nn.ModuleList(row) for row in up_sample])

        final_input_channel = start_feature
        self.final_conv = nn.Conv2d(final_input_channel, out_channel, 1, 1, 0, 1, 1, bias=segout_use_bias)

        # 深度监督
        if self.deep_vision:
            deep_layers = [nn.Conv2d(start_feature, out_channel, 1, bias=segout_use_bias) for _ in
                           range(1, network_depth - 1)]
            self.deep_supervision_layers = nn.ModuleList(deep_layers)

    @staticmethod
    def crop_img(dst, size):
        dst = F.interpolate(dst, size, mode="bilinear", align_corners=False)
        return dst

    def print_net_layers(self):
        for i, lst in enumerate(self.conv_layers):
            print(f"layer: {i}")
            if hasattr(lst, '__iter__') or isinstance(lst, nn.ModuleList):
                for j, block in enumerate(lst):
                    if block and type(block) == _StackedConvLayer:
                        print(f"x({i}, {j}), "
                              f"param: [in: {block.input_channel}, "
                              f"out: {block.output_channel}, "
                              f"dropout rate: {block.dropout_rate}]"
                              )

    def forward(self, x):
        x_results: List[Any] = [[None] * i for i in range(self.network_depth, 0, -1)]
        # 编码器路径（下采样路径）
        current_input = x

        for i in range(self.network_depth):
            # 第0列的处理（最左侧列）
            x_results[i][0] = self.conv_layers[i][0](current_input)
            # 下采样（除了最后一层）
            if i < self.network_depth - 1:
                current_input = self.down_sample[i](x_results[i][0])

        # 解码器路径（上采样和嵌套跳跃连接）
        # 从底部向上，从左到右处理
        for j in range(1, self.network_depth):
            for i in range(self.network_depth - j):

                concat_list = []

                # dense skip
                for k in range(j):
                    feat = x_results[i][k]
                    feat_size = feat.shape[2:]
                    x_size = x_results[i][j - 1].shape[2:]
                    if feat_size != x_size:
                        feat = self.crop_img(feat, x_size)

                    concat_list.append(feat)

                # upsample
                upsampled = self.up_sample[i][j - 1](x_results[i + 1][j - 1])
                upsampled_size = upsampled.shape[2:]
                x_size = x_results[i][j - 1].shape[2:]
                if upsampled_size != x_size:
                    upsampled = self.crop_img(upsampled, x_size)

                concat_list.append(upsampled)

                concatenated = torch.cat(concat_list, dim=1)

                x_results[i][j] = self.conv_layers[i][j](concatenated)

        final_output = self.final_conv(x_results[0][self.network_depth - 1])

        # 收集深度监督输出
        deep_outputs = []
        if self.deep_vision:
            for i, deep_layer in enumerate(self.deep_supervision_layers):
                # 获取对应层的特征图（第0行的第i+1列）
                if i < len(x_results[0]) - 1:
                    deep_output = deep_layer(x_results[0][i + 1])
                    deep_output = self.crop_img(deep_output, final_output.shape[-2:])
                    deep_outputs.append(deep_output)
            return tuple(deep_outputs + [final_output])
        else:
            return final_output

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**cfg['unet++'])

class UnetPlusPlusDenoise(UnetPlusPlusWithLogits):
    def forward(self, x):
        noiseys = super().forward(x)
        return tuple(x - noise for noise in noiseys)


def print_model_params_detailed(model):
    print(f"{'Layer Name':<50} {'Param Count':<15} {'Size (MB)':<10}")
    print("-" * 80)

    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        param_mb = param_count * 4 / 1024 / 1024
        print(f"{name:<50} {param_count:<15,} {param_mb:.4f}")

    print("-" * 80)
    print(f"{'Total':<50} {total_params:<15,} {total_params * 4 / 1024 / 1024:.2f}")

if __name__ == '__main__':
    # m = torch.rand((1, 3, 256, 256))
    config_file = '../configs/net.yaml'
    model = UnetPlusPlusDenoise.from_yaml(config_file)
    # model.print_net_layers()
    print_model_params_detailed(model)
