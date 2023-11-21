import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from dataload.init import getdata
from .network import VAE
import utils.utils as utils
from utils.utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport
from .edg_loss import edge_loss

num_samples=10

def train(config):
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')

    train_dataset, validation_dataset = getdata(config)
    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)
    
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = VAE(in_channels=config.in_ch)
    print('Total params: %.2fM' % (sum(p.numel() for p in gen.parameters())/1000000.0))

    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    gen = gen.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    criterionSoftplus = criterionSoftplus.cuda()
    L1_Loss = nn.SmoothL1Loss(beta=0.5).cuda()

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    print('===> begin')
    start_time=time.time()
    # main
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a_cpu, real_b_cpu, M_cpu = batch[0], batch[1], batch[2]
            real_a = Variable(real_a_cpu).cuda()
            real_b = Variable(real_b_cpu).cuda()
            M = Variable(M_cpu).cuda()

            opt_gen.zero_grad()

            # --- Forward + Backward + Optimize --- #
            (mean, log_var),cloud_removal = gen.forward(real_a)
            # --- Multiple samples --- #
            for i in range(num_samples-1):
                (m, l),c = gen.forward(real_a)
                cloud_removal = cloud_removal + c
                mean = mean + m
                log_var = log_var + l
            
            # --- Take the expectation --- #
            cloud_removal = cloud_removal / num_samples
            mean = mean / num_samples
            log_var = log_var / num_samples
            l1_loss = L1_Loss(cloud_removal, real_b)
            kl_div = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            EDGE_loss = edge_loss(cloud_removal, real_b, 'cuda')
            Loss = l1_loss + 0.01*kl_div + 0.18*EDGE_loss
            Loss.backward()
            opt_gen.step()


        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        if epoch % config.snapshot_interval == 0 and epoch > 50:
            checkpoint(config, epoch, gen)

        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)
