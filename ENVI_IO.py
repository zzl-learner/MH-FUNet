import math
import numpy as np
import re
import os
import torch
from torch import nn
from torch.utils.data import Dataset


def envi_read(root, if_print=True):
    elements = ['samples', 'lines', 'bands', 'data type', 'interleave']
    data_type = {'1': 'u1',
                 '2': 'i2',
                 '3': 'i4',
                 '4': 'f4',
                 '5': 'f8',
                 '12': 'u2',
                 '13': 'u4',
                 '14': 'i8',
                 '15': 'u8'}
    header = {}

    # read header
    rfid = open(root + '.hdr')
    while True:
        strl = rfid.readline()
        if not strl:
            break
        strlist = strl.split('=')
        field = strlist[0].strip()
        if field == elements[0]:
            header['samples'] = int(strlist[1])
        elif field == elements[1]:
            header['lines'] = int(strlist[1])
        elif field == elements[2]:
            header['bands'] = int(strlist[1])
        elif field == elements[3]:
            header['dtype'] = data_type[str(int(strlist[1]))]
        elif field == elements[4]:
            form = strlist[1]
            form = form.replace('\n', '')
            header['format'] = form

    if if_print:
        print('Opening ' + str(header['lines']) + 'rows x ' + str(header['samples']) +
              'cols x ' + str(header['bands']) + 'bands\n')
    # read data

    byte_num = re.findall(r"\d", header['dtype'])
    byte_num = int(byte_num[0])
    if os.path.exists(root + '.dat'):
        fid = open(root + '.dat', 'rb')
    elif os.path.exists(root):
        fid = open(root, 'rb')
    lenth = byte_num * (header['samples']) * (header['lines']) * (header['bands'])
    raw_data = fid.read(lenth)

    if header['format'] == ' bip' or header['format'] == ' BIP':
        image = np.ndarray(shape=(header['lines'], header['samples'], header['bands']),
                           dtype=header['dtype'],
                           buffer=raw_data)

    elif header['format'] == ' bil' or header['format'] == ' BIL':
        image = np.ndarray(shape=(header['lines'], header['bands'], header['samples']),
                           dtype=header['dtype'],
                           buffer=raw_data)
        image = image.swapaxes(1, 2)

    elif header['format'] == ' bsq' or header['format'] == ' BSQ':
        image = np.ndarray(shape=(header['bands'], header['lines'], header['samples']),
                           dtype=header['dtype'],
                           buffer=raw_data)
        image = image.swapaxes(0, 1)
        image = image.swapaxes(1, 2)

    fid.close()
    if if_print:
        print('successfully loaded!\n')

    return image, header


def envi_write(root, img, data_type='float32', interleave='bip', if_print=True):
    p = img.shape
    img = np.ascontiguousarray(img)

    if if_print:
        print('Writing ENVI image ...\n')

    type_all = {'uint8': 1,
                'int16': 2,
                'int32': 3,
                'float32': 4,
                'double': 5,
                'uint16': 12,
                'uint32': 13,
                'int64': 14,
                'uint64': 15}

    # write header
    rfid = open(root + '.hdr', 'w+')
    rfid.write(
        'ENVI\n' + 'description = {}' + '\nsamples = ' + str(p[1]) + '\nlines   = ' + str(p[0]) + '\nbands   = ' + str(
            p[2]) + '\ndata type = ' + str(type_all[data_type]) + '\ninterleave = bip\n')
    rfid.close()

    # write data
    img = img.astype(dtype=data_type)
    fid = open(root + '.dat', 'wb+')

    if interleave == ' bil' or interleave == ' BIL':
        img = img.swapaxes(1, 2)

    elif interleave == ' bsq' or interleave == ' BSQ':
        img = img.swapaxes(1, 2)
        img = img.swapaxes(0, 1)

    fid.write(img)
    fid.close()
    if if_print:
        print('successfully saved!\n')


def get_patch_rate(input, patch_size, rate):
    img_h, img_w = input.shape[1], input.shape[2]
    s = int(patch_size * (1 - rate))
    pad_rows = img_h // s * s + patch_size - img_h
    pad_cols = img_w // s * s + patch_size - img_w
    input = np.pad(input, ((0, 0), (0, pad_rows), (0, pad_cols)), mode='reflect')
    rows = img_h // s + 1
    cols = img_w // s + 1

    patch_num = rows * cols

    output = np.zeros((patch_num, input.shape[0], patch_size, patch_size), dtype='float16')
    count = 0
    for r in range(rows):
        for c in range(cols):
            start_row = r * s
            start_col = c * s
            end_row = start_row + patch_size
            end_col = start_col + patch_size
            output[count] = input[:, start_row:end_row, start_col:end_col]
            count = count + 1
    options = [img_h, img_w, patch_size, s, rows, cols]
    return output, options


def del_patch_new(out_put, options):
    [img_h, img_w, patch_size, s, rows, cols] = options
    input = np.zeros(((rows - 1) * s + patch_size, img_w, out_put.shape[1]), dtype='float32')

    stack_width = patch_size - s
    stack_width_left = stack_width // 2

    stack_height = patch_size - s
    stack_height_upper = stack_height // 2

    stack_in_width = np.zeros((patch_size, (cols - 1) * s + patch_size, out_put.shape[1]), dtype='float32')
    for r in range(rows):
        for c in range(cols):
            start_col = c * s + stack_width_left
            end_col = c * s + patch_size
            if c == 0:
                temp = out_put[r * cols + c][:, :, :]
                stack_in_width[0:patch_size, 0:patch_size, :] = np.transpose(temp, (1, 2, 0))
            else:
                temp = out_put[r * cols + c][:, :, stack_width_left:]
                stack_in_width[0:patch_size, start_col:end_col, :] = np.transpose(temp, (1, 2, 0))
        if r == 0:
            input[0:patch_size, :, :] = stack_in_width[:, 0:img_w, :]
        else:
            start_row = r * s + stack_height_upper
            end_row = r * s + patch_size
            input[start_row:end_row, :, :] = stack_in_width[stack_height_upper:, 0:img_w, :]

    del stack_in_width, temp
    return input[0:img_h, :, :]


def split2patch_envi(in_root, im_name, out_root, patch_size, stack_ratio):
    image, header = envi_read(in_root + im_name)

    im_rows, im_cols, im_bands = image.shape
    L1 = int(patch_size - patch_size * stack_ratio)

    n_cols = int((im_cols - patch_size) // L1)
    n_rows = int((im_rows - patch_size) // L1)

    for i in range(n_rows):
        start_row = i * L1
        end_row = start_row + patch_size
        for j in range(n_cols):
            start_col = j * L1
            end_col = start_col + patch_size
            patch_name = im_name + '_' + str(i) + '_' + str(j)
            envi_write(out_root + patch_name, image[start_row:end_row, start_col:end_col, :], if_print=False)
        patch_name = im_name + '_' + str(i) + '_' + str(j + 1)
        envi_write(out_root + patch_name, image[start_row:end_row, -patch_size:, :], if_print=False)

    for j in range(n_cols):
        start_col = j * L1
        end_col = start_col + patch_size
        patch_name = im_name + '_' + str(i + 1) + '_' + str(j)
        envi_write(out_root + patch_name, image[-patch_size:, start_col:end_col, :], if_print=False)
    patch_name = im_name + '_' + str(i + 1) + '_' + str(j + 1)
    envi_write(out_root + patch_name, image[-patch_size:, -patch_size:, :], if_print=False)

    print('Split successful!\n')


def split2patch_numpy(in_root, im_name, out_root1, out_root2, patch_size, stack_ratio):
    image, header = envi_read(in_root + im_name)

    im_rows, im_cols, im_bands = image.shape
    L1 = int(patch_size - patch_size * stack_ratio)

    n_cols = int((im_cols - patch_size) // L1)
    n_rows = int((im_rows - patch_size) // L1)

    for i in range(n_rows):
        if i < (n_rows - 2):
            start_row = i * L1
            end_row = start_row + patch_size
            for j in range(n_cols):
                start_col = j * L1
                end_col = start_col + patch_size
                patch_name = im_name + '_' + str(i) + '_' + str(j)
                np.save(out_root1 + patch_name, image[start_row:end_row, start_col:end_col, :])
            patch_name = im_name + '_' + str(i) + '_' + str(j + 1)
            np.save(out_root1 + patch_name, image[start_row:end_row, -patch_size:, :])
        else:
            start_row = i * L1
            end_row = start_row + patch_size
            for j in range(n_cols):
                start_col = j * L1
                end_col = start_col + patch_size
                patch_name = im_name + '_' + str(i) + '_' + str(j)
                np.save(out_root2 + patch_name, image[start_row:end_row, start_col:end_col, :])
            patch_name = im_name + '_' + str(i) + '_' + str(j + 1)
            np.save(out_root2 + patch_name, image[start_row:end_row, -patch_size:, :])

    for j in range(n_cols):
        start_col = j * L1
        end_col = start_col + patch_size
        patch_name = im_name + '_' + str(i + 1) + '_' + str(j)
        np.save(out_root2 + patch_name, image[-patch_size:, start_col:end_col, :])
    patch_name = im_name + '_' + str(i + 1) + '_' + str(j + 1)
    np.save(out_root2 + patch_name, image[-patch_size:, -patch_size:, :])

    print('Split successful!\n')


def get_gaussian_kernel(kernel_size=7, sigma=1, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def sub_sampled(input, scale, r, d):
    output = np.transpose(input, (2, 0, 1))
    output = torch.from_numpy(output).type(torch.FloatTensor)
    output = output.unsqueeze(0)
    blur_layer = get_gaussian_kernel(kernel_size=r, sigma=d, channels=output.shape[1])
    output = blur_layer(output)
    down_layer = nn.AdaptiveAvgPool2d((input.shape[0] // scale, input.shape[1] // scale))
    output = down_layer(output)
    output = output.squeeze(0)
    output = output.numpy()
    output = np.transpose(output, (1, 2, 0))
    return output


class ZYDataset(Dataset):
    def __init__(self, root, mode='sim'):
        self.mode = mode
        if mode == 'real':
            self.realHSI_files = np.array([x.path for x in os.scandir(root + '/hsi')])
            self.realMSI_files = np.array([x.path for x in os.scandir(root + '/msi')])
            self.length = len(self.realHSI_files)
        else:
            self.simHSI_files = np.array([x.path for x in os.scandir(root + '/hsi')])
            self.simMSI_files = np.array([x.path for x in os.scandir(root + '/msi')])
            self.simGT_files = np.array([x.path for x in os.scandir(root + '/gt')])
            self.length = len(self.simHSI_files)

    def __getitem__(self, index):
        if self.mode == 'real':
            hsi = np.load(self.realHSI_files[index], allow_pickle=True)
            hsi = np.transpose(hsi, (2, 0, 1))
            hsi = torch.from_numpy(hsi).type(torch.FloatTensor)
            msi = np.load(self.realMSI_files[index], allow_pickle=True)
            msi = np.transpose(msi, (2, 0, 1))
            msi = torch.from_numpy(msi).type(torch.FloatTensor)
            return hsi, msi
        else:
            hsi = np.load(self.simHSI_files[index], allow_pickle=True)
            hsi = np.transpose(hsi, (2, 0, 1))
            hsi = torch.from_numpy(hsi).type(torch.FloatTensor)
            msi = np.load(self.simMSI_files[index], allow_pickle=True)
            msi = np.transpose(msi, (2, 0, 1))
            msi = torch.from_numpy(msi).type(torch.FloatTensor)
            gt = np.load(self.simGT_files[index], allow_pickle=True)
            gt = np.transpose(gt, (2, 0, 1))
            gt = torch.from_numpy(gt).type(torch.FloatTensor)
            return hsi, msi, gt

    def __len__(self):
        return self.length


if __name__ == '__main__':
    ori_root = r'E:/ZY102D/Dataset/' + 'test/' + 'hsi/'
    sub_root = r'E:/ZY102D/Subdataset/' + 'test/' + 'hsi/'
    # root = r'E:\新生培训\Image Fusion\Image Fusion\Data\HyperspecVNIRChikusei\HSI\123'
    # data, p = envi_read(root)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(data[:, :, 1])
    # plt.show()
    # in_root = r'E:/ZY102D/Original/'
    # out_root1 = r'E:/ZY102D/Dataset/train/'
    # out_root2 = r'E:/ZY102D/Dataset/test/'
    # for i in range(1, 8):
    #     split2patch_numpy(in_root + 'hsi/', 'Scene' + str(i), out_root1 + 'hsi/', out_root2 + 'hsi/', 192, 0.3)
    #     split2patch_numpy(in_root + 'msi/', 'Scene' + str(i) + 'msi', out_root1 + 'msi/', out_root2 + 'msi/', 576, 0.3)
    # for x in os.scandir(ori_root):
    #     hsi_ori_res = np.load(x.path)
    #     hsi_sub_res = sub_sampled(hsi_ori_res, 3, 7, 1)
    #     np.save(sub_root + x.name, hsi_sub_res)
    for x in os.scandir(r'E:/ZY102D/Dataset/' + 'train/' + 'hsi/'):
        hsi_ori_res = np.load(x.path)
        np.save(r'E:/ZY102D/Subdataset/' + 'train/' + 'gt/' + x.name, hsi_ori_res)

