
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .rdb3 import RDB
import torch.autograd as autograd
from torch.autograd import Variable
#from dcn.deform_conv import ModulatedDeformConvPack as DCN

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(Discriminator, self).__init__()
        self.layer1_new = nn.Sequential(*[nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, 5, 2, 2)), nn.LeakyReLU(0.2, True)]) # 1/2
        self.layer2 = nn.Sequential(*[nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 5, 2, 2)), nn.LeakyReLU(0.2, True)]) # 1/4
        self.layer3 = nn.Sequential(*[nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2)), nn.LeakyReLU(0.2, True)]) # 1/8
        self.layer4 = nn.Sequential(*[nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 5, 2, 2)), nn.LeakyReLU(0.2, True)]) # 1/16
        self.layer5 = nn.Sequential(*[nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 5, 2, 2)), nn.LeakyReLU(0.2, True)]) # 1/32
        self.att = Self_Attn(ndf * 4, 'relu')
        self.layer6 = nn.Sequential(*[nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 5, 2, 2)), nn.LeakyReLU(0.2, True)]) # 1/64

    def forward(self, input):
        feats = []
        out = self.layer1_new(input)
        feats.append(out)
        out = self.layer2(out)
        feats.append(out)
        out = self.layer3(out)
        feats.append(out)
        out = self.layer4(out)
        feats.append(out)
        out = self.layer5(out)
        feats.append(out)
        out = self.att(out)
        out = self.layer6(out)
        feats.append(out)
        out = out.view(out.size(0), -1)
        # return out, feats
        return out

# --- Downsampling block in GridDehazeNet  --- #
class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out


# --- Upsampling block in GridDehazeNet  --- #
class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out

# --- Main model  --- #
class Generate_quarter(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(Generate_quarter, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)

        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        _in_channels = depth_rate
        for i in range(height - 1):
            for j in range(width // 2):
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):
            for j in range(width // 2, width):
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride

        self.conv1 = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=2, dilation=2)
        self.conv3_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=4, dilation=4)
        self.conv4_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=8, dilation=8)
        self.conv5_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=16, dilation=16)
        self.conv6 = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.offset_conv1 = nn.Conv2d(depth_rate*4, depth_rate*2, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(depth_rate*2, depth_rate*4, 3, 1, 1, bias=True)
        #self.dcnpack = DCN(depth_rate*4, depth_rate*4, 3, stride=1, padding=1, dilation=1, deformable_groups=8, extra_offset_mask=True)
        self.upsamle1 = UpSample(depth_rate*4)
        self.upsamle2 = UpSample(depth_rate*2)

        self.rdb_2_1 = RDB(depth_rate * 2, num_dense_layer, growth_rate)
        self.rdb_1_1 = RDB(depth_rate, num_dense_layer, growth_rate)

    def forward(self, x):
        inp = self.conv_in(x)

        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)

        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])

        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])

        for i in range(1, self.height):
            for j in range(1, self.width // 2):
                channel_num = int(2**(i-1)*self.stride*self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])

        x_index[i][j+1] = self.rdb_module['{}_{}'.format(i, j)](x_index[i][j])
        k = j

        for j in range(self.width // 2 + 1, self.width):
            x_index[i][j] = self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1])

        for i in range(self.height - 2, -1, -1):
            channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
            x_index[i][k+1] = self.coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, k)](x_index[i][k]) + \
                              self.coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())

        for i in range(self.height - 2, -1, -1):
            for j in range(self.width // 2 + 1, self.width):
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())

        out = self.rdb_out(x_index[i][j])

        feat_extra = F.relu(self.conv1(x_index[-1][j]))
        feat_extra = F.relu(self.conv2_atrous(feat_extra))
        feat_extra = F.relu(self.conv3_atrous(feat_extra))
        feat_extra = F.relu(self.conv4_atrous(feat_extra))
        feat_extra = F.relu(self.conv5_atrous(feat_extra))
        feat_extra = F.relu(self.conv6(feat_extra))
        offset = F.relu(self.offset_conv1(feat_extra))
        offset = F.relu(self.offset_conv2(offset))
        #feat_extra = F.relu(self.dcnpack([feat_extra, offset]))
        feat_extra = self.upsamle1(feat_extra, x_index[-2][j].size())
        feat_extra = self.coefficient[-2, 0, 0, :32][None, :, None, None] * x_index[-2][j] + self.coefficient[-2, 0, 0, 32:64][None, :, None, None] * feat_extra
        feat_extra = self.rdb_2_1(feat_extra)
        feat_extra = self.upsamle2(feat_extra, x_index[0][j].size())
        feat_extra = self.coefficient[0, 0, 0, :16][None, :, None, None] * out + self.coefficient[0, 0, 0, 16:32][None, :, None, None] * feat_extra
        out = self.rdb_1_1(feat_extra)
        out = torch.sigmoid(self.conv_out(out))
        #out = out + x
        return 1, out

class Generate_quarter_refine(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(Generate_quarter_refine, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)

        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        _in_channels = depth_rate
        for i in range(height - 1):
            for j in range(width // 2):
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):
            for j in range(width // 2, width):
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride

        self.conv1 = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=2, dilation=2)
        self.conv3_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=4, dilation=4)
        self.conv4_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=8, dilation=8)
        self.conv5_atrous = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=16, dilation=16)
        self.conv6 = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.offset_conv1 = nn.Conv2d(depth_rate*4, depth_rate*2, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(depth_rate*2, depth_rate*4, 3, 1, 1, bias=True)
        #self.dcnpack = DCN(depth_rate*4, depth_rate*4, 3, stride=1, padding=1, dilation=1, deformable_groups=8, extra_offset_mask=True)
        self.upsamle1 = UpSample(depth_rate*4)
        self.upsamle2 = UpSample(depth_rate*2)

        self.rdb_2_1 = RDB(depth_rate * 2, num_dense_layer, growth_rate)
        self.rdb_1_1 = RDB(depth_rate, num_dense_layer, growth_rate)

    def forward(self, x):
        inp = self.conv_in(x)

        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)

        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])

        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])

        for i in range(1, self.height):
            for j in range(1, self.width // 2):
                channel_num = int(2**(i-1)*self.stride*self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])

        x_index[i][j+1] = self.rdb_module['{}_{}'.format(i, j)](x_index[i][j])
        k = j

        for j in range(self.width // 2 + 1, self.width):
            x_index[i][j] = self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1])

        for i in range(self.height - 2, -1, -1):
            channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
            x_index[i][k+1] = self.coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, k)](x_index[i][k]) + \
                              self.coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())

        for i in range(self.height - 2, -1, -1):
            for j in range(self.width // 2 + 1, self.width):
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())

        out = self.rdb_out(x_index[i][j])
        feat_extra = F.relu(self.conv1(x_index[-1][j]))
        feat_extra = F.relu(self.conv2_atrous(feat_extra))
        feat_extra = F.relu(self.conv3_atrous(feat_extra))
        feat_extra = F.relu(self.conv4_atrous(feat_extra))
        feat_extra = F.relu(self.conv5_atrous(feat_extra))
        feat_extra = F.relu(self.conv6(feat_extra))
        offset = F.relu(self.offset_conv1(feat_extra))
        offset = F.relu(self.offset_conv2(offset))
        #feat_extra = F.relu(self.dcnpack([feat_extra, offset]))
        feat_extra = self.upsamle1(feat_extra, x_index[-2][j].size())
        feat_extra = self.coefficient[-2, 0, 0, :32][None, :, None, None] * x_index[-2][j] + self.coefficient[-2, 0, 0, 32:64][None, :, None, None] * feat_extra
        feat_extra = self.rdb_2_1(feat_extra)
        feat_extra = self.upsamle2(feat_extra, x_index[0][j].size())
        feat_extra = self.coefficient[0, 0, 0, :16][None, :, None, None] * out + self.coefficient[0, 0, 0, 16:32][None, :, None, None] * feat_extra
        out = self.rdb_1_1(feat_extra)
        feat = out
        out = F.relu(self.conv_out(out))
        #out = out + x
        return out, feat, feat_extra

class Generate(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(Generate, self).__init__()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate

        self.conv_in_1 = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_in_2 = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_1_downsample = nn.Conv2d(depth_rate * 2, depth_rate * 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride = 2)

        self.conv_2 = nn.Conv2d(depth_rate * 2, depth_rate * 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_2_downsample = nn.Conv2d(depth_rate * 2, depth_rate * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride = 2)

        self.conv_3 = nn.Conv2d(depth_rate * 4, depth_rate * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_3_1 = RDB(depth_rate * 4, num_dense_layer, growth_rate)
        self.rdb_3_2 = RDB(depth_rate * 4, num_dense_layer, growth_rate)

        self.feat_pass = nn.Conv2d(depth_rate, depth_rate * 4, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        self.rdb_3_3 = RDB(depth_rate * 4, num_dense_layer, growth_rate)
        self.rdb_3_4 = RDB(depth_rate * 4, num_dense_layer, growth_rate)
        self.rdb_3_5 = RDB(depth_rate * 4, num_dense_layer, growth_rate)
        self.rdb_3_6 = RDB(depth_rate * 4, num_dense_layer, growth_rate)

        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        self.upsample_L3 = UpSample(depth_rate * 4)

        self.rdb_2_1 = RDB(depth_rate * 2, num_dense_layer, growth_rate)
        self.rdb_2_2 = RDB(depth_rate * 2, num_dense_layer, growth_rate)
        self.rdb_2_3 = RDB(depth_rate * 2, num_dense_layer, growth_rate)
        self.rdb_2_4 = RDB(depth_rate * 2, num_dense_layer, growth_rate)

        self.upsample_L2 = UpSample(depth_rate * 2)

        self.rdb_1_1 = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_1_2 = RDB(depth_rate, num_dense_layer, growth_rate)


    def forward(self, x1, x2, feat):
        inp1 = F.relu(self.conv_in_1(x1))
        inp2 = F.relu(self.conv_in_2(x2))
        conv2 = F.relu(self.conv_1_downsample(torch.cat([inp1, inp2], 1)))
        conv2 = F.relu(self.conv_2(conv2))
        conv3 = F.relu(self.conv_2_downsample(conv2))
        conv3 = F.relu(self.conv_3(conv3))
        conv3 = self.rdb_3_1(conv3)
        conv3 = self.rdb_3_2(conv3)

        # direct
        feat_pass = self.feat_pass(feat)
        conv3 = conv3 + feat_pass
        conv3 = self.rdb_3_3(conv3)
        conv3 = self.rdb_3_4(conv3)
        conv3 = self.rdb_3_5(conv3)
        conv3 = self.rdb_3_6(conv3)
        conv2_up = self.upsample_L3(conv3, conv2.size())
        conv2_up = self.rdb_2_1(conv2_up)
        conv2_up = self.rdb_2_2(conv2_up)
        conv2_up = self.rdb_2_3(conv2_up)
        conv2_up = self.rdb_2_4(conv2_up)
        conv1_up = self.upsample_L2(conv2_up, x1.size())
        conv1_up = self.rdb_1_1(conv1_up)
        conv1_up = self.rdb_1_2(conv1_up)
        out = self.conv_out(conv1_up)
        out = F.relu(out + x2)
        return out

class LossD(nn.Module):
    def __init__(self):
        super(LossD, self).__init__()

    def forward(self, r_x, r_x_hat):
        return (F.relu(1 + r_x_hat) + F.relu(1 - r_x)).mean().reshape(1)

class LossFeat(nn.Module):
    def __init__(self):
        super(LossFeat, self).__init__()

    def forward(self, feats1, feats2):
        loss = []
        for (f1, f2) in zip(feats1, feats2):
            loss.append(F.mse_loss(f1, f2))
        return sum(loss)/len(loss)

class Lap(nn.Module):
    def __init__(self, channels=3):
        super(Lap, self).__init__()
        self.channels = channels
        # print("channels: ", channels.shape)
        kernel = [[0,1,0],[1,-4,1],[0,1,0]]#   [[1,1,1],[1,-8,1],[1,1,1]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)    # (H, W) -> (1, 1, H, W)
        kernel = kernel.expand((int(channels), 1, 3, 3))
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def __call__(self, dehaze, gt):
        #m = nn.Upsample(scale_factor=0.25, mode='nearest')
        #gt = m(gt)
        dehaze = F.conv2d(dehaze, self.weight, padding=1, groups=self.channels)
        gt = F.conv2d(gt, self.weight, padding=1, groups=self.channels)
        loss = []
        for dehaze1, gt1 in zip(dehaze, gt):
            loss.append(F.mse_loss(dehaze1, gt1))
        return sum(loss)/len(loss)
def compute_gradient_penalty(net, real_samples, fake_samples):
  """Calculates the gradient penalty loss for WGAN GP"""
  # Random weight term for interpolation between real and fake samples
  alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=torch.device("cuda:0"))
  # Get random interpolation between real and fake samples
  interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
  d_interpolates = net(interpolates)
#   fake = torch.full((real_samples.size(0),1,8,8 ), 1, device=torch.device("cuda:0"))
  fake=torch.ones_like(d_interpolates)
  # Get gradient w.r.t. interpolates
  gradients = autograd.grad(
    outputs=d_interpolates,
    inputs=interpolates,
    grad_outputs=fake,
    create_graph=True,
    retain_graph=True,
    only_inputs=True,
  )[0]
  gradients = gradients.view(gradients.size(0), -1)
  gradient_penaltys = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
  return gradient_penaltys
if __name__=="__main__":
    net=Discriminator(3)
    a=torch.rand(1,3,128,128)
    # a1=torch.rand(1,3,128,128)
    b=net(a)
    loss=torch.nn.L1Loss()
    print(b.shape)
    # label=torch.zeros((1,1024))
    # print(loss(label,b))