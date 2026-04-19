from typing import List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    batch_2d_kwargs = dict(eps=1e-5, momentum=0.1)
    relu_kwargs = dict(negative_slope=0.1, inplace=False)

    def __init__(self, in_channel, out_channel, conv_kernel_size,
                 dropout_rate: Optional[int] = 0.1,
                 stride=None, dropout_in_uplayer=False,
                 activation='leakyrelu_norm2d'):
        super().__init__()

        if stride is not None:
            self.conv_2d_kwargs['stride'] = stride

        if dropout_rate is not None:
            self.dropout_kwargs['p'] = dropout_rate

        self.conv2d = nn.Conv2d(in_channel, out_channel, conv_kernel_size, **self.conv_2d_kwargs)
        self.dropout = nn.Dropout(**self.dropout_kwargs) if (dropout_in_uplayer and 
                                                             self.dropout_kwargs['p'] > 0.0) else None
        self.batch2d = nn.BatchNorm2d(out_channel, **self.batch_2d_kwargs)
        self.nonlin = nn.LeakyReLU(**self.relu_kwargs)
        self.relu = nn.ReLU()
        self.mode = activation

    def forward(self, x):
        global CONV_NUM
        CONV_NUM += 1

        x = self.conv2d(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.mode == ACTIVATION[0]:
            return self.batch2d(self.nonlin(x))
        elif self.mode == ACTIVATION[1]:
            return self.nonlin(self.batch2d(x))
        elif self.mode == ACTIVATION[2]:
            return self.batch2d(self.relu(x))

        return F.relu(x)


class _StackedConvLayer(nn.Module):
    def __init__(self, in_feature_channel: int, out_feature_channel: int, stack_size, conv_kernel_size,
                 dropout_rate=None,
                 dropout_in_uplayer=False, first_stride=None):
        super().__init__()
        self.input_channel = in_feature_channel
        self.output_channel = out_feature_channel

        self.conv_blocks = nn.Sequential(
            *([_SingleConv2D(in_feature_channel, out_feature_channel, conv_kernel_size, dropout_rate,
                             stride=first_stride,
                             dropout_in_uplayer=dropout_in_uplayer)] +
              [_SingleConv2D(out_feature_channel, out_feature_channel, conv_kernel_size, dropout_rate, stride=None,
                             dropout_in_uplayer=dropout_in_uplayer) for _ in range(stack_size - 1)])
        )
        self.dropout_rate = dropout_rate

    def forward(self, x):
        return self.conv_blocks(x)


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
            # print(x.shape)
            skips.append(x)
            x = self.pool_blocks[i](x)
            # print(f"Down layer shape: {x.shape}")
        return x, skips


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='blinear', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)


# No dropout
class _UpLayer(nn.Module):
    # up_features = [512, 256, 128, 64]

    def __init__(self, in_channel: int, start_feature: int, feature_count: int, stack_size: int,
                 conv_kernel_size: tuple, dropout_rate: Optional[int], *, cat_crop_transform=CenterCrop,
                 stack_op=_StackedConvLayer, up_conv=nn.ConvTranspose2d,
                 convolutional_upsampling=False, **kwargs):
        super().__init__()
        self.up_conv_blocks, self.conv_blocks = nn.ModuleList(), nn.ModuleList()
        self.up_features = [start_feature // (2 ** i) for i in range(max(1, feature_count))]
        for i, feature in enumerate(self.up_features):
            up_conv_method = up_conv(in_channel, feature, (2, 2), 2, 0, bias=False)

            self.up_conv_blocks.append(up_conv_method)
            self.conv_blocks.append(
                stack_op(in_channel, feature, stack_size, conv_kernel_size, dropout_rate, dropout_in_uplayer=True)
            )
            in_channel = feature

        self.cat_crop_transform = cat_crop_transform

        self.convolutional_upsampling = convolutional_upsampling

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

            if not self.convolutional_upsampling:
                skip_feature = skips[len(self.up_features) - i - 1]

                # skip connection
                x = self.crop_and_concat(x, skip_feature)

            x = self.conv_blocks[i](x)
            # print(f"Up layer shape: {x.shape}")
        return x


class ExtendableUnet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, stack_size=2, down_layer_start_feature=64, dropout_rate=None,
                 network_depth=5, segout_use_bias=False, act_last=False, pool_kernel_size=(2, 2),
                 conv_kernel_size=(3, 3)):
        super(ExtendableUnet, self).__init__()

        # 1, Left down layer
        self.down = _DownLayer(in_channel,
                               down_layer_start_feature,
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
                           dropout_rate)

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


class UnetDenoiseResidual(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, stack_size=2, down_layer_start_feature=64, dropout_rate=None,
                 network_depth=5, segout_use_bias=False, pool_kernel_size=(2, 2), conv_kernel_size=(3, 3), **kwargs):
        super().__init__()
        self.base_unet = ExtendableUnet(in_channel=in_channel, out_channel=out_channel, stack_size=stack_size,
                                        down_layer_start_feature=down_layer_start_feature, dropout_rate=dropout_rate,
                                        network_depth=network_depth, segout_use_bias=segout_use_bias,
                                        pool_kernel_size=pool_kernel_size,
                                        conv_kernel_size=conv_kernel_size)

    def forward(self, x):
        y_noise = self.base_unet(x)
        clean = x - y_noise
        return clean


class UnetWithResidual(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, stack_size=2, down_layer_start_feature=64, negative_slope=0.1,
                 network_depth=5, segout_use_bias=False, require_1x1_conv=True,
                 pool_kernel_size=(2, 2), conv_kernel_size=(3, 3), **kwargs):
        super().__init__()
        self.base_unet = ExtendableUnet(in_channel=in_channel, out_channel=out_channel, stack_size=stack_size,
                                        down_layer_start_feature=down_layer_start_feature, network_depth=network_depth,
                                        segout_use_bias=segout_use_bias, pool_kernel_size=pool_kernel_size,
                                        conv_kernel_size=conv_kernel_size)
        self.conv2d = nn.Conv2d(out_channel, out_channel, conv_kernel_size, padding=1)
        self.bn2d = nn.BatchNorm2d(out_channel)

        # 我们需要负值信息，保留负值
        self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
        self.require_1x1_conv = require_1x1_conv

        if self.require_1x1_conv:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)

    def create_residual_layer(self, nums):
        for i in range(nums):
            yield nn.Sequential(
                self.conv2d,
                self.bn2d,
                self.relu,
                self.conv2d,
                self.bn2d,
            )

    def forward(self, x):
        x = self.base_unet(x)

        # Residual
        res_iter = self.create_residual_layer(2)
        y, temp_x = None, x

        for layer in res_iter:
            y = layer(temp_x)
            if self.require_1x1_conv:
                temp_x = self.conv(temp_x)
            y += temp_x
            temp_x = y

        return y


class UnetPlusPlusWithLogits(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, network_depth=5, first_stride=1, stack_size=2, dropout_rate=0.1,
                 start_feature=64, segout_use_bias=False, pool_kernel_size=(2, 2), conv_kernel_size=(3, 3),
                 transpose_conv_kernel_size=(2, 2), deep_vision=True, dropout_in_upconv=True,
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
            current_input_channel = in_channels if i == 0 else feature_per_layer // 2

            for j in range(network_depth - i):
                # 创建卷积块
                if j == 0:
                    in_ch = current_input_channel
                    if i == 0:
                        conv_layers[i][j] = stack_op(in_ch, feature_per_layer, stack_size, conv_kernel_size,
                                                     first_stride=first_stride)
                    else:
                        conv_layers[i][j] = stack_op(in_ch, feature_per_layer, stack_size, conv_kernel_size)
                else:
                    in_ch = feature_per_layer * (j + 1)
                    conv_layers[i][j] = stack_op(in_ch, feature_per_layer, stack_size, conv_kernel_size, dropout_rate,
                                                 dropout_in_uplayer=dropout_in_upconv)

                # 创建上采样层（除了最后一列）
                if j < network_depth - i - 1:
                    up_sample[i][j] = transpose_conv_op(
                        start_feature * (2 ** (i + 1)), feature_per_layer,
                        transpose_conv_kernel_size, stride=2, padding=0, bias=False
                    )

            # 创建下采样层（除了最后一层）
            if i < network_depth - 1:
                down_sample[i] = pool_op(pool_kernel_size)

        # 转换为ModuleList以便正确注册参数
        self.conv_layers = nn.ModuleList([nn.ModuleList(row) for row in conv_layers])
        self.down_sample = nn.ModuleList(down_sample)
        self.up_sample = nn.ModuleList([nn.ModuleList(row) for row in up_sample])

        final_input_channel = start_feature
        self.final_conv = nn.Conv2d(final_input_channel, out_channels, 1, 1, 0, 1, 1, bias=segout_use_bias)
        # TODO:深度监督
        if self.deep_vision:
            deep_layers = [nn.Conv2d(start_feature, out_channels, 1, bias=segout_use_bias) for _ in
                           range(1, network_depth - 1)]
            self.deep_supervision_layers = nn.ModuleList(deep_layers)

    @staticmethod
    def crop_img(dst, size):
        dst = F.interpolate(dst, size, mode="bilinear", align_corners=False)
        return dst

    def print_net_layers(self):
        for i, lst in enumerate(self.conv_layers):
            print(f"layer: {i}")
            if hasattr(lst, '__iter__'):
                for j, block in enumerate(lst):
                    if block and isinstance(block, _StackedConvLayer):
                        print(f"x({i}, {j}), "
                              f"param: [in: {block.input_channel}, out: {block.output_channel}, dropout rate: {block.dropout_rate}]")

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


class UnetPlusPlusDenoise(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, network_depth=5, first_stride=1, stack_size=2, dropout_rate=0.1,
                 start_feature=64, segout_use_bias=False, pool_kernel_size=(2, 2), conv_kernel_size=(3, 3),
                 transpose_conv_kernel_size=(2, 2), deep_vision=True, dropout_in_upconv=True, up_sample_crop=Upsample,
                 stack_op=_StackedConvLayer, pool_op=nn.MaxPool2d, transpose_conv_op=nn.ConvTranspose2d, **kwargs):
        super().__init__()
        self.nested_unet = UnetPlusPlusWithLogits(in_channels=in_channel, out_channels=out_channel,
                                                  network_depth=network_depth,
                                                  first_stride=first_stride, stack_size=stack_size,
                                                  dropout_rate=dropout_rate, start_feature=start_feature,
                                                  segout_use_bias=segout_use_bias, pool_kernel_size=pool_kernel_size,
                                                  conv_kernel_size=conv_kernel_size,
                                                  transpose_conv_kernel_size=transpose_conv_kernel_size,
                                                  deep_vision=deep_vision,
                                                  dropout_in_upconv=dropout_in_upconv, up_sample_crop=up_sample_crop,
                                                  stack_op=stack_op, pool_op=pool_op,
                                                  transpose_conv_op=transpose_conv_op)

    def print_net_layers(self):
        self.nested_unet.print_net_layers()

    def forward(self, x):
        noiseys = self.nested_unet(x)
        return tuple(x - noise for noise in noiseys)


if __name__ == '__main__':
    m = torch.rand((1, 1, 256, 256))
    model = UnetWithResidual(1, 1, network_depth=2, dropout_rate=0.1)
    logits = model(m)
    print(logits)

    # print(m[0].squeeze().permute((2, 1, 0)))
