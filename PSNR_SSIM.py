import cv2
import numpy as np
import os
import torch
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim


def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def torchSSIM(tar_img, prd_img):
    return ssim(tar_img, prd_img, data_range=1.0, size_average=True)


def ComputePSNR_SSIM(img_dir,gt_path):
    error_list_ssim, error_list_psnr , error_list_uiqm= [],[],[]
    for dir_path in img_dir:
        enhanced_name = dir_path.split('\\')[-1]
        gt_name = enhanced_name
        enhanced = cv2.imread(dir_path)
        gt = cv2.imread(os.path.join(gt_path, gt_name))
        enhanced = cv2.resize(enhanced, [256, 256])
        gt = cv2.resize(gt, [256, 256])
        gt = TF.to_tensor(gt)
        enhanced = TF.to_tensor(enhanced)
        error_psnr = torchPSNR(gt, enhanced)
        gt = gt.unsqueeze(0)
        enhanced = enhanced.unsqueeze(0)
        error_ssim = torchSSIM(gt,enhanced)
        print(enhanced_name, error_psnr, error_ssim)
        error_list_psnr.append(error_psnr)
        error_list_ssim.append(error_ssim)

    return np.array(error_list_ssim), np.array(error_list_psnr)

if __name__=='__main__':
    enhanced_path = r'data/result'
    gt_path=r'data/gt'
    img_name = os.listdir(enhanced_path)
    img_dir = [ os.path.join(enhanced_path,name) for name in img_name]
    ssims,psnrs = ComputePSNR_SSIM(img_dir,gt_path)
    print ("SSIM >> Mean: {:.3f} std: {:.3f}".format(np.mean(ssims), np.std(ssims)))
    print ("PSNR >> Mean: {:.3f} std: {:.3f}".format(np.mean(psnrs), np.std(psnrs)))

