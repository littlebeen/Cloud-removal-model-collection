import os
import cv2
import random
import numpy as np

import torch
from torch.backends import cudnn
from PIL import Image

def gpu_manage(config):
    if config.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.gpu_ids))
        config.gpu_ids = list(range(len(config.gpu_ids)))

    # print(os.environ['CUDA_VISIBLE_DEVICES'])

    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    print('Random Seed: ', config.manualSeed)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not config.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def save_image(out_dir, x, num, epoch, filename=None):
    img = (x*255).clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img.contiguous().cpu()[0]
    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path = os.path.join(test_dir, filename+'.png')
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.png'.format(num))

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    Image.fromarray(np.uint8(img)).save(test_path)


def save_imagenir(out_dir, x, num, epoch, filename=None):
    img = (x*255).clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1)[0]
    w,h,c =img.shape
    nir = img[:,:,3]
    nir = nir.expand(3, w,h)
    nir = nir.permute( 1, 2, 0)
    img = img.contiguous().cpu()
    nir = nir.contiguous().cpu()
    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path = os.path.join(test_dir, filename+'.png')
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.png'.format(num))
    if filename is not None:
        test_path_nir= os.path.join(test_dir, filename+'nir.png')
    else:
        test_path_nir = os.path.join(test_dir, 'test_{0:04d}nir.png'.format(num))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    Image.fromarray(np.uint8(img[:,:,0:3])).save(test_path)
    Image.fromarray(np.uint8(nir)).save(test_path_nir)

def checkpoint(config, epoch, gen, dis=None):
    model_dir = os.path.join(config.out_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    net_gen_model_out_path = os.path.join(model_dir, 'gen_model_epoch_{}.pth'.format(epoch))
    torch.save(gen.state_dict(), net_gen_model_out_path)
    if(dis):
         net_dis_model_out_path = os.path.join(model_dir, 'dis_model_epoch_{}.pth'.format(epoch))
         torch.save(dis.state_dict(), net_dis_model_out_path)
    print("Checkpoint saved to {}".format(model_dir))


def make_manager():
    if not os.path.exists('.job'):
        os.makedirs('.job')
        with open('.job/job.txt', 'w', encoding='UTF-8') as f:
            f.write('0')


def job_increment():
    with open('.job/job.txt', 'r', encoding='UTF-8') as f:
        n_job = f.read()
        n_job = int(n_job)
    with open('.job/job.txt', 'w', encoding='UTF-8') as f:
        f.write(str(n_job + 1))
    
    return n_job

def heatmap(img):
    if len(img.shape) == 3:
        b,h,w = img.shape
        heat = np.zeros((b,3,h,w)).astype('uint8')
        for i in range(b):
            heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,:,:],cv2.COLORMAP_JET),(2,0,1))
    else:
        b,c,h,w = img.shape
        heat = np.zeros((b,3,h,w)).astype('uint8')
        for i in range(b):
            heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,0,:,:],cv2.COLORMAP_JET),(2,0,1))
    return heat

def save_attention_as_heatmap(filename, att):
    att_heat = heatmap(att)
    cv2.imwrite(filename, att_heat)
    print(filename, 'saved')