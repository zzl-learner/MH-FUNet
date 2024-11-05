import numpy as np
def get_patch_rate(input, patch_size, rate):
    img_h, img_w = input.shape[1], input.shape[2]
    s = int(patch_size * (1 - rate))
    pad_rows = img_h//s * s + patch_size - img_h
    pad_cols = img_w//s * s + patch_size - img_w
    input = np.pad(input, ((0, 0), (0, pad_rows), (0, pad_cols)), mode = 'reflect')
    rows = img_h//s + 1
    cols = img_w//s + 1

    patch_num = rows * cols

    output = np.zeros((patch_num, input.shape[0], patch_size, patch_size))
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
import struct
def readenvi(root):
    elements = ['samples ', 'lines   ', 'bands   ', 'data type ', 'interleave ']
    d = ['uchar', 'int16', 'int32', 'float32', 'float64', 'uint16', 'uint32', 'int64', 'uint64']
    p=[]
    rfid = open(root+'.hdr')
    while True:
        strl = rfid.readline()
        if not strl:
            break
        strlist = strl.split('=')
        if strlist[0] == elements[0]:
            p.append(int(strlist[1]))
        elif strlist[0] == elements[1]:
            p.append(int(strlist[1]))
        elif strlist[0] == elements[2]:
            p.append(int(strlist[1]))
        elif strlist[0] == elements[4]:
            c = str(strlist[1])
        elif strlist[0] == elements[3]:
            t = strlist[1]
            t = int(t)
    swap = p[0]
    p[0] = p[1]
    p[1] = swap
    if c == ' bil\n':
        print('Opening ' + str(p[0]) + 'rows x ' + str(p[1]) + 'cols x ' + str(p[2]) + 'bands')
        fid = open(root + '.dat', 'rb')
        code = str(p[1]) + 'f'
        i = 0
        image = np.zeros(p, dtype='float16')
        while True:
            strb = fid.read(t * p[1])
            if strb == b"":
                break
            line = np.array(struct.unpack(code, strb))
            j = i % p[2]
            k = i // p[2]
            image[k, :, j] = line
            i = i + 1
    if c == ' bip\n':
        print('Opening ' + str(p[0]) + 'rows x ' + str(p[1]) + 'cols x ' + str(p[2]) + 'bands')
        fid = open(root + '.dat', 'rb')
        code = str(p[2]) + 'f'
        i = 0
        image = np.zeros(p, dtype='float16')
        while True:
            strb = fid.read(t * p[2])
            if strb == b"":
                break
            point = np.array(struct.unpack(code, strb))
            j = i % p[1]
            k = i // p[1]
            image[k, j, :] = point
            i = i + 1
    if c == ' bsq\n':
        print('Opening ' + str(p[0]) + 'rows x ' + str(p[1]) + 'cols x ' + str(p[2]) + 'bands')
        fid = open(root + '.dat', 'rb')

        code = str(p[0]*p[1]) + 'f'
        i = 0
        image = np.zeros(p, dtype='float32')
        while True:
            strb = fid.read(t * p[0] * p[1])
            if strb == b"":
                break
            point = np.array(struct.unpack(code, strb))
            image[:, :, i] = point.reshape((p[0],p[1]))
            i = i + 1
    return image, p
def del_patch(out_put, options):
    [img_h, img_w, patch_size, s, rows, cols] = options
    input = np.zeros(((rows-1)*s+patch_size, (cols-1)*s+patch_size, out_put.shape[1]), dtype='float32')
    mid = (patch_size-s)//2
    count = 0
    for c in range(rows):
        for r in range(cols):
            start_row = r * s
            start_col = c * s
            end_row = start_row + patch_size
            end_col = start_col + patch_size
            if r==0 or c==0:
                input[start_col:end_col,start_row:end_row, :] = np.transpose(out_put[count], (1,2,0))
            else:
                input[start_col + mid:end_col, start_row + mid:end_row, :] = np.transpose(out_put[count][:, mid:patch_size, mid:patch_size], (1,2,0))
            count = count + 1
    return input[0:img_h, 0:img_w, :]

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

def writeenvi(root, img):
    elements={'samples =' 'lines   =' 'bands   =' 'data type ='}
    d = ['uchar', 'int16', 'int32', 'float32', 'float64', 'uint16', 'uint32', 'int64', 'uint64']
    p = img.shape
    print('Writing ENVI image ...')
    rfid = open(root+'.hdr', 'w+')
    rfid.write('ENVI\n'+'description = {}'+'\nsamples = '+str(p[1])+'\nlines   = '+str(p[0])+'\nbands   = '+str(p[2])+'\ndata type = '+str(4)+'\ninterleave = bip\n')
    rfid.close()
    img = img.astype('float32')
    fid = open(root+'.dat', 'wb+')
    code = str(p[0])+'f'
    i = 0
    fid.write(img)
    fid.close()

if __name__ == '__main__':
    gt, p = readenvi('F:\code\HyperspectraFusion\Methods\work_dir\Hysurecaveexamplesgt')
    pre, p = readenvi('F:\code\HyperspectraFusion\Methods\work_dir\Hysurecaveexamplesim')
    name = ['2msi', '1', '2', '2msi', '3', '3msi', '4', '4msi', '5', '5msi', '6', '6msi', '7', '7msi']
    root = 'F:\code\HyperspectraFusion\DataSet\ZY102D\Scene'
    #saveroot = 'E:\ZY1-02D-reflect\SceneOne\Scene'
    for i in name:
        file = root+i
        #save = saveroot+i
        image, p = readenvi(file)
        patch_image, option = get_patch_rate(image, 120, 0.1)
        out_img = del_patch(patch_image, option)
        import matplotlib.pyplot as plt
        t = out_img[:,:,0].astype('float32')
        plt.imshow(t)
        plt.show()
        print('finish')
        #writeenvi(save, image)
        #print('save file: '+save)