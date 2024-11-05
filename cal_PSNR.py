import numpy as np
def cal_psnr(outputs, gt):
    batch_psnr = 0.0
    del_list = []
    del_list = set(del_list)
    for k in range(gt.shape[0]):
        mse = np.zeros(gt.shape[1])
        psnr = np.zeros(gt.shape[1])
        for i in range(gt.shape[1]):
            im1 = outputs[k, i, :, :].detach().cpu().numpy()
            im2 = gt[k, i, :, :].detach().cpu().numpy()
            mse[i] = ((im1 - im2) ** 2).mean()
            if i in del_list:
                psnr[i] = 0
            else:
                psnr[i] = 10 * np.log10(1 / mse[i])
        batch_psnr += np.sum(psnr)/(psnr.__len__()-del_list.__len__())
    return batch_psnr/gt.shape[0]