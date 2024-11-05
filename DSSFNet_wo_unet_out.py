from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
# from DSSFNet_withgrad_model import DSSFNet_with_grad
from DSSFmodel_wo_unet import DSSFNet_with_grad



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


'''------------------------------------------outputimg model----------------------------------------------------'''
import tools.patch_tools as tool
import numpy as np
import ENVI_IO
import gc

def read_and_divide(img_path, patch_size, stack_rate):
    img_o, p = ENVI_IO.envi_read(img_path)
    img = np.transpose(img_o, (2, 0, 1))
    img, option = ENVI_IO.get_patch_rate(img, patch_size, stack_rate)
    # img = img.astype('float16')
    return img, p, option

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")
    # device = torch.device("cpu")
    hsi, p1, option1 = read_and_divide(r'/home/data2/zzl22/fusion/Dataset/ZY102D/Scene5', 80, 0.1)
    msi, p2, option2 = read_and_divide(r'/home/data2/zzl22/fusion/Dataset/ZY102D/Scene5msi', 240, 0.1)
    print('finish load data')
    

    model = DSSFNet_with_grad(148, 8, 3, 10)
    # model = nn.DataParallel(model)
    model_root = '/home/data2/zzl22/fusion/Methods/work_dir/DSSFNet_wo_unet_20240131_server_L1loss_5e-5_no5'

    model_path = model_root + '/epoch839.pth'
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, False)
    model.train()
    model = model.to(device)
    print('Model load Done!')
    allimg = np.zeros((hsi.shape[0], hsi.shape[1], msi.shape[2], msi.shape[3]),
                          dtype='float32')

    patch_row = option2[4]
    patch_col = option2[5]

    btc_size = 12
    num_all = patch_col * patch_row
    s1 = 'hello'
    if btc_size > num_all:
        print('The number of image patch in one test iteration should not be larger than the total one!')

    btc_num = num_all // btc_size
    btc_remain = num_all - btc_num * btc_size
    with torch.no_grad():
        for btc_iter in range(btc_num):
            index = range(btc_iter * btc_size, (btc_iter + 1) * btc_size, 1)
            HSI, MSI = torch.tensor(hsi[index]), torch.tensor(msi[index])
            # HSI, MSI = HSI.unsqueeze(dim=0), MSI.unsqueeze(dim=0)
            HSI, MSI = HSI.type(torch.FloatTensor).to(device), MSI.type(torch.FloatTensor).to(device)
    
            outputs1 = model(HSI, MSI)
            outputs1 = outputs1.squeeze(dim=0)
            # im = outputs1[2,:,:,:].detach().cpu().numpy().transpose(1,2,0)
            # ENVI_IO.envi_write(model_root + '/scene5_test', im)
            allimg[index] = outputs1.detach().cpu().numpy()
            del HSI, MSI, outputs1
            gc.collect()
    
            if btc_remain:
                btc_iter = btc_iter + 1
                index = range(btc_iter * btc_size, btc_iter * btc_size + btc_remain, 1)
                HSI, MSI = torch.tensor(hsi[index]), torch.tensor(msi[index])
                # HSI, MSI = HSI.unsqueeze(dim=0), MSI.unsqueeze(dim=0)
                HSI, MSI = HSI.type(torch.FloatTensor).to(device), MSI.type(torch.FloatTensor).to(device)
        
                outputs1 = model(HSI, MSI)
        
        
                outputs1 = outputs1.squeeze(dim=0)
                allimg[index] = outputs1.detach().cpu().numpy()
                del HSI, MSI, outputs1
                gc.collect()
    
    allimg = ENVI_IO.del_patch_new(allimg, option2)
    ENVI_IO.envi_write(model_root + '/scene5_20240131', allimg)
