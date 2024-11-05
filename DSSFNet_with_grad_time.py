from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from DSSFmodel import DSSFNet_with_grad



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

from torch.utils.data import DataLoader
import time
import os

'''if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda")
    dataset = ZYFDataset(r'/home/data2/zzl22/DataSet/ZY102D/train')
    trainloader = DataLoader(dataset, batch_size=10, shuffle=True)
    epochs = 200
    model = DSSFNet_with_grad(148, 8, 3, 10)
    model = nn.DataParallel(model)
    model_root = './work_dir/DSSFNet_with_Grad_20230404'

    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    my_loss = torch.nn.SmoothL1Loss()
    save_interval = 20

    isExists = os.path.exists(model_root)
    if not isExists:
        os.makedirs(model_root)
        print(model_root + ' create success')
    else:
        print(model_root + ' has existed')

    for epoch in range(epochs):
        model_path = model_root + '/epoch' + str(epoch) + '.pth'
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            HSI, MSI, GT = data
            HSI, MSI, GT = HSI.type(torch.FloatTensor).to(device), MSI.type(torch.FloatTensor).to(device), GT.type(
                torch.FloatTensor).to(device)
            optimizer.zero_grad()
            outputs = model(HSI, MSI)
            loss = my_loss(outputs, GT)
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
    print('PyCharm')'''

from tools.cal_PSNR import cal_psnr
from tools.cal_SSIM import cal_ssim
from tools.cal_SAM import cal_sam
from tools.cal_ERGAS import cal_ergas
import tools.patch_tools as tool
import numpy as np
import ENVI_IO


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device("cuda")
    dataset = ZYFDataset(r'/home/data2/zzl22/fusion/Dataset/ZY102D_new/test')

    testloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    epochs = 200

    model = DSSFNet_with_grad(148, 8, 3, 10)
    # model = nn.DataParallel(model)

    model_root = './work_dir/DSSFNet_with_Grad_20230505_server_L1loss_1e-4_no5'

    model_path = model_root + '/epoch359.pth'
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, False)
    # load_state_dict(new_state_dict)
    model.eval()
    model = model.to(device)
    print('Model load Done!')
    running_psnr = 0.0
    running_ssim = 0.0
    running_sam = 0.0
    running_ergas = 0.0
    time_c = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            HSI, MSI, GT = data
            HSI, MSI, GT = HSI.type(torch.FloatTensor).to(device), MSI.type(torch.FloatTensor).to(device), GT.type(
                torch.FloatTensor).to(device)
            time1 = time.time()
            # HSI = F.interpolate(HSI, scale_factor=3, mode='bicubic')
            outputs = model(HSI, MSI)
            time2 = time.time()
            time_c += time2 - time1
    print(time_c)
            
            
