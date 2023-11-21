import os
import shutil
import yaml
from attrdict import AttrMap
import utils.utils as utils
from utils.utils import gpu_manage, save_image, checkpoint
from dataload.init import getdata
from torch.utils.data import DataLoader
import torch
from eval import test,testcomp
#from thop import profile
import time
import torch.optim as optim
import numpy as np

if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    os.makedirs(config.out_dir,exist_ok=True)

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')
    train_dataset, validation_dataset = getdata(config)
    print('validation dataset:', len(validation_dataset))
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)
    
    ### MODELS LOAD ###
    print('===> Loading models')
    if(config.model=='spagan'):
        from model.models_spagan.gen.SPANet import Generator,SPANet
        gen = Generator(gpu_ids=config.gpu_ids,channel =config.in_ch)
    if(config.model=='amgan'):
        from model.models_amgan_cr.gen.AMGAN import Generator,AMGAN
        gen = Generator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)
    if(config.model=='mdsa'):
        from model.msda.model import Generate_quarter
        gen = Generate_quarter(in_channels=config.in_ch,height=3,width=6,num_dense_layer=4,growth_rate=16)
    if(config.model=='mn'):
        from model.mn.model import MPRNet
        gen = MPRNet(in_c=config.in_ch, out_c=config.in_ch)
    if(config.model=='cvae'):
        from model.cvae.network import VAE
        gen = VAE(in_channels=config.in_ch)
    
    gen=gen.cuda()

    # input = torch.randn(2, 3, 256, 256).cuda()
    # all_time=[]
    # code to show the Computational complexity (speed, parameters,memory,complexity(GFLOPs))
    # for _ in range(100):
    #     time_start = time.time()
    #     predict = gen(input)
    #     time_end = time.time()
    #     all_time.append(time_end-time_start)
    # print('Speed: %.5f\n' % (1/np.mean(all_time)))
    #flops, params = profile(gen, inputs=(input, ))
    # print('Complexity: %.3fM' % (flops/1000000000), end=' GFLOPs\n')
    # optimizer = optim.SGD(gen.parameters(), lr=0.9, momentum=0.9, weight_decay=0.0005)
    # for _ in range(1000):
    #     optimizer.zero_grad()
    #     gen(input)

    param = torch.load('./pre_train/'+os.listdir('./pre_train/')[0])
    gen.load_state_dict(param)
    print('load {} as pretrained model'.format(os.listdir('./pre_train/')[0]))
    criterionMSE = torch.nn.MSELoss()
    with torch.no_grad():
        log_validation = test(config, validation_data_loader, gen, criterionMSE, 200)