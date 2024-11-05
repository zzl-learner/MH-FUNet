import numpy as np
def cal_ergas(outputs, gt, scale):
    batch_ergas = 0.0
    del_list = []
    del_list = set(del_list)
    for k in range(gt.shape[0]):
        a = np.zeros(gt.shape[1])
        for i in range(gt.shape[1]):
            im1 = outputs[k, i, :, :].detach().cpu().numpy()
            im2 = gt[k, i, :, :].detach().cpu().numpy()
            if i in del_list:
                a[i] = 0
            else:
                a[i] = ((im1 - im2) ** 2).mean()/(im2.mean()**2)
        batch_ergas += (100/scale)*np.sqrt(np.sum(a)/(a.__len__()-del_list.__len__()))
    return batch_ergas/gt.shape[0]