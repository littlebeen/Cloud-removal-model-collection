##################自己方法     七个波段 #####################

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from ..layers import CBR
from ..models_utils import weights_init, print_network

#### attentive-recurrent network替换前半部分

# Set iteration time
ITERATION = 4

###### Layer
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)



class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        # 三层卷积
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out

###### Network
class AMGAN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AMGAN, self).__init__()

        self.conv_in = nn.Sequential(

            conv3x3(in_ch, 32),
            nn.ReLU(True)
        )
        self.conv_ins = nn.Sequential(

            conv3x3(in_ch + 1, 32),
            nn.ReLU(True)
        )
        self.res_block4 = Bottleneck(32,32)
        self.res_block5 = Bottleneck(32,32)
        self.res_block6 = Bottleneck(32,32)
        self.det_conv0 = nn.Sequential(
            nn.Conv2d(in_ch + 1, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )

        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
        )

        #获得mask之后的处理
        self.res_block1 = Bottleneck(32, 32)
        self.res_block2 = Bottleneck(32, 32)
        self.res_block3 = Bottleneck(32, 32)
        self.res_block4 = Bottleneck(32, 32)
        self.res_block5 = Bottleneck(32, 32)
        self.res_block6 = Bottleneck(32, 32)

        self.conv_out = nn.Sequential(
            conv3x3(32, out_ch)
        )

    def forward(self, input):
        input = input.type(torch.cuda.FloatTensor)  # 转Float

        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        mask = Variable(torch.ones(batch_size, 1, row, col)).cuda() / 2.
        h = Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, 32, row, col)).cuda()

        mask_list = []
        # 4次循环
        for i in range(ITERATION):
            x = torch.cat((input, mask), 1)
            x = self.det_conv0(x)
            resx = x

            # 5个residual block
            x = F.relu(self.det_conv1(x) + resx)
            resx = x
            x = F.relu(self.det_conv2(x) + resx)
            resx = x
            x = F.relu(self.det_conv3(x) + resx)
            resx = x
            x = F.relu(self.det_conv4(x) + resx)
            resx = x
            x = F.relu(self.det_conv5(x) + resx)
            x = torch.cat((x, h), 1)  #上一次LATM 的结果

            # LSTM的4步
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)

            c = f * c + i * g
            h = o * torch.tanh(c)

            # 1个卷积
            mask = self.det_conv_mask(h)

            mask_list.append(mask)

        # input 和 mask 的连接   由7变成8
        x = torch.cat((input, mask), 1)

        x = self.conv_ins(x)

        out = self.conv_in(input)

        out = self.res_block1(out)

        out = F.relu(self.res_block1(out) * mask_list[-1] + x)
        out = F.relu(self.res_block2(out) * mask_list[-1] + x)
        out = F.relu(self.res_block3(out) * mask_list[-1] + x)

        # 2个RB
        out = F.relu(self.res_block4(out) + out)
        out = F.relu(self.res_block5(out) + out)

        # 一个卷积
        out = self.conv_out(out)

        # Attention4——attention map
        # out——结果
        return mask_list[-1], out


class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids

        self.gen = nn.Sequential(OrderedDict([('gen', AMGAN(in_ch, out_ch))]))

        self.gen.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            # 多个GPU，并行
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            # 一个
            return self.gen(x)

