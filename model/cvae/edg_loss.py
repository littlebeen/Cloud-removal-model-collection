import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# --- Edge loss function  --- #
class Edg_Capture(nn.Module):
    def __init__(self):
        super(Edg_Capture, self).__init__()
        kernel = [[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=1)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

def edge_loss(x,y,device):
    laplace = Edg_Capture().to(device)
    L1 = nn.L1Loss().to(device)
    out = L1(laplace(x),laplace(y))
    return out
