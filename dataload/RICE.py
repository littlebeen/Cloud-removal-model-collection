import numpy as np
import os
from torch.utils import data
from utils.imgproc import imresize
import skimage.io as io
import blobfile as bf


class TrainDataset(data.Dataset):

    def __init__(self, config, isTrain=True):
        super().__init__()
        self.config = config
        if(isTrain):
            self.datasets_dir = '/xxxx/' + config.datasets_dir+'/train' #change to the path of your dataset
        else:
            self.datasets_dir = '/xxx/' + config.datasets_dir+'/test'
        self.imlist = sorted(bf.listdir(self.datasets_dir+'/label'))

    def __getitem__(self, index):
        t = io.imread(os.path.join(self.datasets_dir, 'label', str(self.imlist[index]))).astype(np.float32)
        x = io.imread(os.path.join(self.datasets_dir, 'cloud', str(self.imlist[index]))).astype(np.float32)
        t = imresize(t, 1/2)  #resize your image
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