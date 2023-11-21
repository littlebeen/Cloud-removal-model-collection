import glob
import cv2
import random
import numpy as np
import pickle
import os
from PIL import Image
from torch.utils import data
from utils.imgproc import imresize
import skimage.io as io
class TrainDataset(data.Dataset):

    def __init__(self, config, isTrain=True):
        super().__init__()
        self.config = config
        if(isTrain):
            self.datasets_dir = '/media/lscsc/nas/jialu/data/' + config.datasets_dir +'/train'
            train_list_file = os.path.join(self.datasets_dir, 'train.txt')
            self.imlist = np.loadtxt(train_list_file, str)
        else:
            self.datasets_dir = '/media/lscsc/nas/jialu/data/' + config.datasets_dir +'/test'
            val_list_file = os.path.join(self.datasets_dir, 'test.txt')
            self.imlist = np.loadtxt(val_list_file, str)

    def __getitem__(self, index):
        t = io.imread(os.path.join(self.datasets_dir, 'label', str(self.imlist[index]))).astype(np.float32)
        x = io.imread(os.path.join(self.datasets_dir, 'cloud', str(self.imlist[index]))).astype(np.float32)

        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)

        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        filename = self.imlist[index].split('.')[0]

        return x, t, M,filename

    def __len__(self):
        return len(self.imlist)