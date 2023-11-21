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
import math
import blobfile as bf
def save_image(out_dir, x, num, epoch, filename=None):
    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path = os.path.join(test_dir, filename+'.png')
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.png'.format(num))

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    Image.fromarray(np.uint8(x)).save(test_path)


class TrainDataset(data.Dataset):

    def __init__(self, config, isTrain=True):
        super().__init__()
        self.config = config
        self.datasets_dir = '/media/lscsc/nas/jialu/CloudRemoval/data/RICE_DATASET/' + config.datasets_dir
        if(isTrain):
            self.datasets_dir = '/media/lscsc/nas/jialu/data/' + config.datasets_dir+'/train'
        else:
            self.datasets_dir = '/media/lscsc/nas/jialu/data/' + config.datasets_dir+'/test'
        self.imlist = sorted(bf.listdir(self.datasets_dir+'/label'))

    def __getitem__(self, index):
        t = io.imread(os.path.join(self.datasets_dir, 'label', str(self.imlist[index]))).astype(np.float32)
        x = io.imread(os.path.join(self.datasets_dir, 'cloud', str(self.imlist[index]))).astype(np.float32)
        t = imresize(t, 1/2)
        x = imresize(x, 1/2)


        M = np.clip((t-x).sum(axis=2), 0, 10).astype(np.float32)

        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        filename = self.imlist[index].split('.')[0]

        return x, t, M,filename

    def __len__(self):
        return len(self.imlist)