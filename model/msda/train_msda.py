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
from model.msda.model import Generate_quarter,compute_gradient_penalty,Discriminator,Lap
import utils.utils as utils
from utils.utils import gpu_manage, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport
from .perceptual import LossNetwork
from torchvision.models import vgg16

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

    gen = Generate_quarter(height=3,width=6,num_dense_layer=4,growth_rate=16, in_channels = config.in_ch)
    print('Total params: %.2fM' % (sum(p.numel() for p in gen.parameters())/1000000.0))


    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    dis = Discriminator(input_nc=config.in_ch)

    if config.dis_init is not None:
        param = torch.load(config.dis_init)
        dis.load_state_dict(param)
        print('load {} as pretrained model'.format(config.dis_init))

    vgg_model=vgg16(pretrained=True).features[:16]
    vgg_model=vgg_model.to('cuda')
    for param in vgg_model.parameters():
        param.requires_grad=False
    loss_network=LossNetwork(vgg_model)
    loss_network.eval()
    loss_rec1=nn.SmoothL1Loss()
    loss_lap=Lap()

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    gen = gen.cuda()
    dis = dis.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    criterionSoftplus = criterionSoftplus.cuda()

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
            opt_gen.zero_grad()
            M = Variable(M_cpu).cuda()
            att, fake_b = gen.forward(real_a)

            # decloud_1,feat_extra_1=net(cloud)
            rec_loss1=loss_rec1(fake_b,real_b)
            perceptual_loss=loss_network(fake_b,real_b)
            lap_loss=loss_lap(fake_b,real_b)
            loss_gen=rec_loss1*1.2+0.04*perceptual_loss

            D_out1=dis(fake_b)
            loss_D=-torch.mean(D_out1)
            
            loss_total=loss_D*0.05+loss_gen
            loss_total.backward()
            opt_gen.step()
            #generator finished 

            #discriminator start
            for param in dis.parameters():
                param.requires_grad=True
            fake_b=fake_b.detach()

            opt_dis.zero_grad()
            D_out1=dis(fake_b)
            # loss_D1=bce_loss(D_out1,Variable(torch.FloatTensor(D_out1.data.size()).fill_(GEN_label)).to(device))
            loss_D1_fake=torch.mean(D_out1)
            # loss_D1=loss_D1/2
            # loss_D1.backward()
            # loss_D_value=loss_D1
            D_out2=dis(real_b)
            # loss_D1=bce_loss(D_out2,Variable(torch.FloatTensor(D_out2.data.size()).fill_(ORI_label)).to(device))
            loss_D2_real=-torch.mean(D_out2)
            gp=compute_gradient_penalty(dis,real_b,fake_b)
            # loss_D1=loss_D1/2
            errD=loss_D1_fake+loss_D2_real+gp
            errD.backward()
            # loss_D_value=loss_D_value+loss_D1
            opt_dis.step()

            # log
            if iteration % config.print_every == 0:
                print("===> Epoch[{}]({}/{}): loss_d_fake: {:.4f} loss_d_real: {:.4f} loss_g_gan: {:.4f} loss_g_l1: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_D1_fake.item(), loss_D2_real.item(), loss_D.item(), loss_gen.item()))
                
                log = {}
                log['epoch'] = epoch
                log['iteration'] = len(training_data_loader) * (epoch-1) + iteration
                log['gen/loss'] = loss_total.item()
                log['dis/loss'] = errD.item()

                logreport(log)

        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        if epoch % config.snapshot_interval == 0 and epoch > 50:
            checkpoint(config, epoch, gen, dis)

        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)
