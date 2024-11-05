import numpy as np
def cal_sam(outputs, gt):
    batch_sam = 0.0
    for k in range(gt.shape[0]):
        sam = np.zeros(gt.shape[2]*gt.shape[3])
        for i in range(gt.shape[2]):
            for j in range(gt.shape[3]):
                im1 = outputs[k, :, i, j].detach().cpu().numpy().reshape((1, gt.shape[1]))
                im2 = gt[k, :, i, j].detach().cpu().numpy().reshape((gt.shape[1], 1))
                s = np.sum(np.dot(im1, im2))
                t = np.sqrt(np.sum(im1 ** 2)) * np.sqrt(np.sum(im2 ** 2))
                if t == 0:
                    sam[i * gt.shape[3] + j] = 0
                else:
                    sam[i*gt.shape[3]+j] = np.arccos(s/t)
        batch_sam += np.mean(sam)
    return (batch_sam/gt.shape[0])*180/3.1415926