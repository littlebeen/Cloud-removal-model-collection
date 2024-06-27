import numpy as np
from skimage.metrics import structural_similarity as SSIM
from torch.autograd import Variable
import lpips
from utils.utils import save_image,save_imagenir
import torch
from skimage.metrics import peak_signal_noise_ratio as PSNR

loss_fn = lpips.LPIPS(net='alex', version=0.1)

def caculate_lpips(img0,img1):
    im1=np.copy(img0.cpu().numpy())
    im2=np.copy(img1.cpu().numpy())
    im1=torch.from_numpy(im1.astype(np.float32))
    im2 = torch.from_numpy(im2.astype(np.float32))
    im1.unsqueeze_(0)
    im2.unsqueeze_(0)
    current_lpips_distance  = loss_fn.forward(im1, im2)
    return current_lpips_distance 

def caculate_ssim(imgA, imgB):
    imgA1 = np.tensordot(imgA.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    imgB1 = np.tensordot(imgB.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    score = SSIM(imgA1, imgB1, data_range=255)
    return score

def caculate_psnr( imgA, imgB):
    imgA1 = imgA.cpu().numpy().transpose(1, 2, 0)
    imgB1 = imgB.cpu().numpy().transpose(1, 2, 0)
    psnr = PSNR(imgA1, imgB1, data_range=255)
    return psnr

def get_image_arr(dataset):  #the id of the image you want to save 
    if(dataset=='RICE1'):
        return ['0','105','143','368','425','458','495']
    elif(dataset=='RICE2'):
        return ['49','17','185','209','309','619','408','630']
    elif(dataset=='T-Cloud'):
        return ['278','142','162','449','930','1261','1652']
    elif(dataset=='My14' or dataset=='My24'):
        return ['4','5','7','8','36','37','65']
    else:
        return []


def test(config, test_data_loader, gen, criterionMSE, epoch):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    for i, batch in enumerate(test_data_loader):
        x, t, filename = Variable(batch[0]), Variable(batch[1]),batch[3]
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)

        att, out = gen(x)
        if epoch % config.snapshot_interval == 0 and epoch > 20 and filename[0] in get_image_arr(config.datasets_dir):
            if(x.shape[1]==3):
                save_image(config.out_dir, x, i, epoch, filename=filename[0]+'Cloudy')
                save_image(config.out_dir, t, i, epoch, filename=filename[0]+'GT')
                save_image(config.out_dir, out, i, epoch, filename=filename[0]+'CR')
            else:  #it handle the multispectral nir layer (the situation that image contain 4 band RGB and nir)
                save_imagenir(config.out_dir, x, i, epoch, filename=filename[0]+'Cloudy')
                save_imagenir(config.out_dir, t, i, epoch, filename=filename[0]+'GT')
                save_imagenir(config.out_dir, out, i, epoch, filename=filename[0]+'CR')

        imgA = (out[0]*255).clamp(0, 255).to(torch.uint8)
        imgB = (t[0]*255).clamp(0, 255).to(torch.uint8)
        psnr = caculate_psnr(imgA, imgB)
        c,w,h=imgA.shape
        if(imgA.shape[0]==4):
            lpips=0
            ssim=0
            for i in range(imgA.shape[0]):
                imA = imgA[i]
                imA = imA.expand(3,w,h)
                imB = imgB[i]
                imB = imB.expand(3,w,h)
                ssim1 = caculate_ssim(imA, imB)
                lpips1 = caculate_lpips(imA, imB)
                ssim+=ssim1
                lpips+=lpips1
            ssim=ssim/imgA.shape[0]
            lpips=lpips/imgA.shape[0]
        else:
            ssim = caculate_ssim(imgA, imgB)
            lpips = caculate_lpips(imgA, imgB)
        avg_psnr += psnr
        avg_ssim += ssim
        avg_lpips += lpips
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    avg_lpips = avg_lpips / len(test_data_loader)

    print("===> Avg. PSNR: {:.3f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))
    print("===> Avg. Lpips: {:.4f} dB".format(avg_lpips.item()))
    
    log_test = {}
    log_test['epoch'] = epoch
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim

    return log_test
