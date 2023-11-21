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
            self.datasets_dir = '/xxx/' + config.datasets_dir+'/train' #change to the path of your dataset
        else:
            self.datasets_dir = '/xxx/' + config.datasets_dir+'/test'

        self.imlistl = sorted(bf.listdir(self.datasets_dir+'/label'))

    def __getitem__(self, index):
        # a dataset contain 4 bands. it read the nir band and RGB band separately
        t = io.imread(os.path.join(self.datasets_dir, 'label', str(self.imlistl[index]))).astype(np.float32)
        x = io.imread(os.path.join(self.datasets_dir, 'cloud', str(self.imlistl[index]))).astype(np.float32)
        nirt = io.imread(os.path.join(self.datasets_dir, 'nir/label', str(self.imlistl[index]))).astype(np.float32)[:,:,0]
        nirx = io.imread(os.path.join(self.datasets_dir, 'nir/cloud', str(self.imlistl[index]))).astype(np.float32)[:,:,0]
        t =np.concatenate([t,nirt[:,:,np.newaxis]],axis=2)
        x =np.concatenate([x,nirx[:,:,np.newaxis]],axis=2)
        t = imresize(t, 1/2)
        x = imresize(x, 1/2)

        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
        #M = io.imread(os.path.join(self.datasets_dir, 'mask', str(self.imlistl[index]))).astype(np.float32)
        # M[M>0.5]=1
        # M[M<=0.5]=0
        
        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        filename = self.imlistl[index].split('.')[0]

        return x, t, M, filename

    def __len__(self):
        return len(self.imlistl)
