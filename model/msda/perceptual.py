
# --- Imports --- #
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2


# --- Perceptual loss network  --- #
class LossNetworkF(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetworkF, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            size = dehaze_feature.size()
            pad = torch.zeros(size).cuda()
            print(size)
            import ff
            #dehaze_feature = torch.unsqueeze(dehaze_feature, 2)
            dehaze_feature = torch.cat([dehaze_feature,pad], dim=2)
            #gt_feature = torch.unsqueeze(gt_feature, 2)
            gt_feature = torch.cat([gt_feature,pad], dim=2)
            f1 = torch.fft(dehaze_feature, 2)
            f2 = torch.fft(gt_feature, 2)
            loss.append(F.mse_loss(f1[:,:,0]*f1[:,:,0]+f1[:,:,1]*f1[:,:,1], f2[:,:,0]*f2[:,:,0]+f2[:,:,1]*f2[:,:,1]))

        return torch.from_numpy(sum(loss)/len(loss)).cuda()

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)

class LossNetworkL1(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetworkL1, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        self.loss_rec = nn.L1Loss()

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(self.loss_rec(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)

