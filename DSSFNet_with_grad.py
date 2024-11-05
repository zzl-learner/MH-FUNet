
# -*- coding: UTF-8 -*-

from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from torch.utils.data import DataLoader
import time
from tools.cal_PSNR import cal_psnr
from tools.cal_SSIM import cal_ssim
from tools.cal_SAM import cal_sam
from tools.cal_ERGAS import cal_ergas
from DSSFmodel import DSSFNet_with_grad
from DSSFmodel import Get_gradient


class ZYFDataset(Dataset):
    def __init__(self, root, mode='sim'):
        self.mode = mode
        if mode == 'real':
            self.realHSI_files = np.array([x.path for x in os.scandir(root + '/realHSI')])
            self.realMSI_files = np.array([x.path for x in os.scandir(root + '/realMSI')])
            self.realPAN_files = np.array([x.path for x in os.scandir(root + '/realPAN')])
            self.length = len(self.realHSI_files)
        else:
            self.simHSI_files = np.array([x.path for x in os.scandir(root + '/simHSI')])
            self.simMSI_files = np.array([x.path for x in os.scandir(root + '/simMSI')])
            self.simGT_files = np.array([x.path for x in os.scandir(root + '/simGT')])
            self.length = len(self.simHSI_files)

    def __getitem__(self, index):
        if self.mode == 'real':
            hsi = np.load(self.realHSI_files[index], allow_pickle=True)
            hsi = np.transpose(hsi, (2, 0, 1))
            hsi = torch.from_numpy(hsi).type(torch.FloatTensor)
            msi = np.load(self.realMSI_files[index], allow_pickle=True)
            msi = np.transpose(msi, (2, 0, 1))
            msi = torch.from_numpy(msi).type(torch.FloatTensor)
            pan = np.load(self.realPAN_files[index], allow_pickle=True)
            pan = np.transpose(pan, (2, 0, 1))
            pan = torch.from_numpy(pan).type(torch.FloatTensor)
            return hsi, msi, pan
        else:
            hsi = np.load(self.simHSI_files[index], allow_pickle=True)
            # hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())
            hsi = np.transpose(hsi, (2, 0, 1))
            hsi = torch.from_numpy(hsi).type(torch.FloatTensor)
            msi = np.load(self.simMSI_files[index], allow_pickle=True)
            # msi = (msi - msi.min()) / (msi.max() - msi.min())
            msi = np.transpose(msi, (2, 0, 1))
            msi = torch.from_numpy(msi).type(torch.FloatTensor)
            gt = np.load(self.simGT_files[index], allow_pickle=True)
            # gt = (gt-gt.min())/(gt.max()-gt.min())
            gt = np.transpose(gt, (2, 0, 1))
            gt = torch.from_numpy(gt).type(torch.FloatTensor)
            return hsi, msi, gt

    def __len__(self):
        return self.length

from math import exp



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM_LOSS(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM_LOSS, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MEAN_SAMLoss(torch.nn.Module):
    def __init__(self):
        super(MEAN_SAMLoss, self).__init__()

    def forward(self, im, gt):
        batch_sam = 0.0
        for k in range(gt.shape[0]):
            im1 = im[k, :, :, :].squeeze(0)
            im2 = gt[k, :, :, :].squeeze(0)
            corr = torch.sum(torch.mul(im1, im2), 0)
            i1 = torch.sum(torch.mul(im1, im1), 0)
            i2 = torch.sum(torch.mul(im2, im2), 0)
            para = torch.mul(i1, i2)
            the = torch.arccos(corr / (para + 1e-6))
            # the = corr/(para + 1e-6)
            the = torch.mean(torch.mean(the, 0))
            batch_sam = batch_sam + the
        return (batch_sam / gt.shape[0])# * 180 / 3.1415926



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
    device = torch.device("cuda")
    dataset = ZYFDataset(r'/home/data2/zzl22/DataSet/ZY102D_new/train')
    trainloader = DataLoader(dataset, batch_size=12, shuffle=True)
    epochs = 1000
    model = DSSFNet_with_grad(148, 8, 3, 10)
    model = nn.DataParallel(model)
    model_root = './work_dir/DSSFNet_with_Grad_20230523_server_L1loss_5e-5_no5'

    model.train()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
    my_loss1 = nn.SmoothL1Loss()
    my_loss2 = nn.MSELoss()
    loss_module1 = Get_gradient(148).to(device)
    loss_module2 = Get_gradient(8).to(device)
    my_loss3 = SSIM_LOSS()
    my_loss4 = MEAN_SAMLoss().to(device)
    save_interval = 10

    isExists = os.path.exists(model_root)
    if not isExists:
        os.makedirs(model_root)
        print(model_root + ' create success')
    else:
        print(model_root + ' has existed')
        
    lamda1 = torch.linspace(start=1, end=0.001, steps=epochs)
    lamda2 = torch.linspace(start=0.01, end=1, steps=epochs)

    for epoch in range(epochs):
        model_path = model_root + '/epoch' + str(epoch) + '.pth'
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            HSI, MSI, GT = data
            HSI, MSI, GT = HSI.type(torch.FloatTensor).to(device), MSI.type(torch.FloatTensor).to(device), GT.type(
                torch.FloatTensor).to(device)
            optimizer.zero_grad()
            outputs = model(HSI, MSI)
            # loss = lamda2[epoch] * (my_loss4(outputs, GT)) + lamda1[epoch] * my_loss2(loss_module1(outputs), loss_module2(MSI))
        
            loss = my_loss1(outputs, GT)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2 == 1:
                print(time.strftime('%Y-%m-%d %H:%M:%S  ',
                                    time.localtime(time.time())) + '[epoch : %d, batch: %4d] loss: %.6f ' %
                      (epoch + 1, i + 1, running_loss / i))
                
        with open(model_root + '/history.txt', 'a+') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' epoch:' + str(
                epoch) + ',loss:' + str((running_loss / i)) + '\n')
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), model_path)
            print("model weight saved as", model_path)
        if (epoch + 1) % 5 == 0:
            im_psnr = cal_psnr(outputs, GT)
            im_ssim = cal_ssim(outputs, GT)
            im_sam = cal_sam(outputs, GT)
            im_ergas = cal_ergas(outputs, GT, 8)
            print(im_psnr, im_ssim.item(), im_sam, im_ergas)
    print('PyCharm')
