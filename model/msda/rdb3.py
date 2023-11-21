#Edited by Weikang YU
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, dilation = 1):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=dilation, dilation=dilation)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class AdaIn(nn.Module):
    def __init__(self):
        super(AdaIn, self).__init__()
        self.eps = 1e-5

    def forward(self, x, mean_style, std_style):
        B, C, H, W = x.shape

        feature = x.view(B, C, -1)

        #print (mean_feat.shape, std_feat.shape, mean_style.shape, std_style.shape)
        std_style = std_style.view(B, C, 1)
        mean_style = mean_style.view(B, C, 1)
        adain = std_style * (feature) + mean_style

        adain = adain.view(B, C, H, W)
        return adain

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)

# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate, dilations = [1, 1, 1, 1]):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate, dilation = dilations[i]))
            _in_channels += growth_rate   
        self.residual_dense_layers = nn.Sequential(*modules)

        _in_channels_rt=in_channels
        rt_modules = []
        for i in range(num_dense_layer):
            rt_modules.append(MakeDense(_in_channels_rt, growth_rate, dilation = dilations[i]))
            _in_channels_rt += growth_rate   
        self.rt_residual_dense_layers = nn.Sequential(*rt_modules)

        self.conv_1x1_a = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_1x1_c = nn.Conv2d(_in_channels_rt, in_channels, kernel_size=1, padding=0)

        _in_channels_no_style = in_channels
        no_style_modules = []
        for i in range(num_dense_layer):
            no_style_modules.append(MakeDense(_in_channels_no_style, growth_rate))
            _in_channels_no_style += growth_rate

        self.residual_dense_layers_no_style = nn.Sequential(*no_style_modules)
        self.conv_1x1_b = nn.Conv2d(_in_channels_no_style, in_channels, kernel_size=1, padding=0)

        self.norm = nn.InstanceNorm2d(in_channels)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.adaIn = AdaIn()

        self.global_feat = nn.AdaptiveAvgPool2d((1, 1))
        self.global_feat_a = nn.AdaptiveAvgPool2d((1, 1))
        self.style = nn.Linear(in_channels // 2, in_channels * 2)
        self.conv_1x1_style = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv_1x1_style_rt = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv_a=nn.Conv2d(in_channels,in_channels,3,padding=1)
        #T gamma and beta
        self.conv_gamma_t = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)
        self.conv_beta_t = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)

        #R gamma and beta
        self.conv_gamma_r = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)
        self.conv_beta_r = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)


        self.conv_att_A = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.conv_att_B = nn.Conv2d(in_channels, 1, 3, padding=1)
        # self.conv_att_C = nn.Conv2d(in_channels, 1, 3, padding=1)


        self.conv_r=nn.Conv2d(in_channels,1,3,padding=1)
        self.in_channels = in_channels

        self.conv_1x1_final = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0)

        self.coefficient = nn.Parameter(torch.Tensor(np.ones((1, 2))), requires_grad=True)
        self.ca = CALayer(in_channels)
        self.pool = nn.AvgPool2d((7, 7), stride=(1, 1), padding=(3, 3))

        #self.noise = ApplyNoise(in_channels)

    def forward(self, x):
        # residual
        bottle_feat = self.residual_dense_layers(x)
        out = self.conv_1x1_a(bottle_feat)
        out = out + x

        # base residual， self-guieded learn mean，std，gamma，and beta
        style_feat_1 = F.relu(self.conv_1x1_style(out))
        style_feat_2 = F.relu(self.conv_1x1_style_rt(out))

        style_feat = self.global_feat(style_feat_1)

        style_feat = torch.flatten(style_feat, start_dim = 1)
        
        style_feat = self.style(style_feat)
        
        # mean, std
        style_mean = style_feat[:, :self.in_channels] #mean and std is shape 1*1*2 each channel
        style_std = style_feat[:, self.in_channels:]

        # rt_bottle_feat=self.rt_residual_dense_layers(x)
        # rt_out=self.conv_1x1_c(rt_bottle_feat)
        # out_rt=rt_out+x
        # style_feat_1_rt=F.relu(self.conv_1x1_c(out_rt))

        gamma_r = self.conv_gamma_r(style_feat_1)
        beta_r = self.conv_beta_r(style_feat_1)

        gamma_t = self.conv_gamma_t(style_feat_2)
        beta_t = self.conv_beta_t(style_feat_2)

        """
        gamma beta 以及 style mean style std都是用于计算G和L
        在除云的应用当中，根据公式只需要1个I即可
        总体而言，除雾的公式为I(x)=t(x)J(x)* (1-t(x))A(x)
        而除云的公式为I(x)=aIr(x,y)t(x,y)+I(1-t(x,y))
        """
        y = self.norm(x)
        out_no_style = self.residual_dense_layers_no_style(y)
        out_no_style = self.conv_1x1_b(out_no_style)
        out_no_style = y + out_no_style
        



        #out_no_style = self.noise(out_no_style, None)
        out_no_style = self.norm2(out_no_style)
        out_att_A = torch.sigmoid(self.conv_att_A(out_no_style))
        out_att_B = torch.sigmoid(self.conv_att_B(out_no_style))
        # out_att_C = torch.sigmoid(self.conv_att_C(out_no_style))
        out_new_style = self.adaIn(out_no_style, style_mean , style_std) #G
        out_new_gamma_T = out_no_style * (1 + gamma_t) + beta_t #T
        out_new_gamma_R = out_no_style * (1 + gamma_r) + beta_r #R
        out_new=out_new_gamma_R*out_att_A+(out_new_gamma_T*(1-out_att_B)+out_new_style*out_att_B)*(1-out_att_A)
        # out_new = out_att * out_new_style + (1 - out_att) * out_new_gamma
        out = self.conv_1x1_final(torch.cat([out, out_new], dim = 1))
        out = self.ca(out)
        out = out + x
        return out

if __name__ == "__main__":
    model=RDB(16,4,16)
    a=torch.rand(4,16,128,128)
    b=model(a)
    print(b.shape)