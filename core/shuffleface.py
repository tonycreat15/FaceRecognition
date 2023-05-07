import torch
import torch.nn as nn
import math
import numpy as np

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

# http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.PReLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.PReLU(),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

class ShuffleFaceNet(nn.Module):
    def __init__(self, stages_repeats=[4, 8, 4], stages_out_channels=[24, 176, 352, 704, 1024], inverted_residual=InvertedResidual):
        super(ShuffleFaceNet, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(),
        )
        input_channels = output_channels

        self.gdc = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=1, padding=0, bias=False, groups=input_channels),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(),
        )

        input_channels = output_channels
        output_channels = 128

        self.linearconv = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(output_channels)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = nn.functional.interpolate(x, size=[112, 112])
        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        #x = x.mean([2, 3])  # globalpool 
        x = self.gdc(x)
        # x = np.squeeze(x, axis=2)
        x = x.view(x.size(0), 1024, 1)
        x = self.linearconv(x)
        x = x.view(x.size(0), 128, 1, 1)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        out = self._forward_impl(x)
        norm = torch.norm(out, 2, 1, True)
        output = torch.div(out, norm)
        return output, norm

if __name__ == "__main__":
    # input = Variable(torch.FloatTensor(2, 3, 112, 96))
    net = ShuffleFaceNet()
    print(net)
    # x = net(input)
    # print(x.shape)