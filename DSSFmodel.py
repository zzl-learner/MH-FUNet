from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Get_gradient(nn.Module):
    def __init__(self, bands):
        super(Get_gradient, self).__init__()
        kernel_v = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).repeat(bands, 1, 1, 1)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).repeat(bands, 1, 1, 1)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        self.b = bands

    def forward(self, x):
        x_v = F.conv2d(x, self.weight_v, stride=1, padding=1, groups=self.b)
        x_h = F.conv2d(x, self.weight_h, stride=1, padding=1, groups=self.b)

        # x1 = torch.cat(torch.sqrt(torch.pow(x_v, 2) + 1e-6), torch.sqrt(torch.pow(x_h, 2) + 1e-6), dim=1)
        # x1 = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        x1 = torch.cat((torch.sum(x_v, 1).unsqueeze(1), torch.sum(x_h, 1).unsqueeze(1)), dim=1)
        # x1 = torch.max(x1, 1).values.unsqueeze(1)
        # x1 = torch.sum(x1, 1).unsqueeze(1)
        return x1


class Resblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Resblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(128, output_channels, kernel_size=3, stride=1, padding=1)
        )
        self.afun = nn.PReLU()

    def forward(self, x):
        xt = self.conv(x)
        x = x + xt
        x = self.afun(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            Resblock(128, 128),
            Resblock(128, 128),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, output_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpConvx2(nn.Module):
    def __init__(self, num_channels):
        super(UpConvx2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class DownConvx2(nn.Module):
    def __init__(self, num_channels):
        super(DownConvx2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class downSample(nn.Module):
    def __init__(self, num_channels, scale):
        super(downSample, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=2 * scale + 1, stride=scale, padding=scale),
        )

    def forward(self, X):
        X = self.layers(X)
        return X


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.register_buffer()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class flow_estimate(nn.Module):
    def __init__(self, hsi_channels, msi_channels):
        super(flow_estimate, self).__init__()
        self.grad_h = Get_gradient(hsi_channels)
        self.fea_h = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fea_m = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.grad_m = Get_gradient(msi_channels)
        self.est = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )

    def forward(self, hsi, msi):
        hsi_f = self.fea_h(self.grad_h(hsi))
        msi_f = self.fea_m(self.grad_m(msi))
        flow = self.est(torch.cat((hsi_f, msi_f), 1))
        return flow


class fusionunit(nn.Module):
    def __init__(self, hsi_channels, msi_channels, scale):
        super(fusionunit, self).__init__()
        self.CAM = ChannelAttention(hsi_channels)
        self.SAM = SpatialAttention()
        self.feture1 = ResNet(hsi_channels, 128)
        self.feture2 = ResNet(msi_channels, 128)
        self.convin = DoubleConv(256, 128)
        self.conv2 = DoubleConv(256, 128)
        self.down1 = nn.Sequential(
            DownConvx2(128),
            DoubleConv(128, 128)
        )
        self.down2 = nn.Sequential(
            DownConvx2(128),
            DoubleConv(128, 128)
        )
        self.down3 = nn.Sequential(
            DownConvx2(128),
            DoubleConv(128, 128)
        )
        self.up1 = UpConvx2(128)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = UpConvx2(128)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = UpConvx2(128)
        self.conv3 = DoubleConv(256, 128)
        self.convout = nn.Sequential(
            nn.Conv2d(128, hsi_channels, kernel_size=3, stride=1, padding=1),
        )
        self.scale = scale
        self.flow_est = flow_estimate(128, 128)

    def forward(self, Y, Z):
        Yt = F.interpolate(Y, scale_factor=self.scale, mode='bicubic')
        Yt = self.feture1(Yt)
        Zt = self.feture2(Z)
        flow = self.flow_est(Yt, Zt)
        B, C, H, W = Zt.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.cuda()
        vgrid = Variable(grid) + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        # 取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0  # 取出光流u这个维度，同上
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(Zt, vgrid, align_corners=True, mode='bilinear')
        Yt = output
        
        X1 = torch.cat((Yt, Zt), 1)
        X1 = self.convin(X1)
        X2 = self.down1(X1)
        X3 = self.down2(X2)
        X4 = self.down3(X3)
        Xout3 = self.up1(X4)
        Xout3 = torch.cat((Xout3, X3), 1)
        Xout3 = self.conv1(Xout3)
        Xout2 = self.up2(Xout3)
        Xout2 = torch.cat((Xout2, X2), 1)
        Xout2 = self.conv2(Xout2)
        Xout1 = self.up3(Xout2)
        Xout1 = torch.cat((Xout1, X1), 1)
        Xout1 = self.conv3(Xout1)
        Xout = self.convout(Xout1)
        ca = self.CAM(Y)
        sa = self.SAM(Z)
        # ca = self.CAM(Yt)
        # sa = self.SAM(Zt)
        Xt = torch.mul(Xout, ca)
        Xt = torch.mul(Xt, sa)
        Xout = Xout + Xt
        return Xout


class IRU(nn.Module):
    def __init__(self, num_channels, num_msic, scale):
        super(IRU, self).__init__()
        # self.E = nn.Conv2d(num_endmember,num_channels,kernel_size=1, padding=0)
        self.MR = nn.Conv2d(num_channels, num_msic, kernel_size=3, stride=1, padding=1)
        self.MD = downSample(num_channels, scale)
        self.afun1 = nn.LeakyReLU(0.2, inplace=True)
        self.afun2 = nn.LeakyReLU(0.2, inplace=True)
        self.fusion = fusionunit(num_channels, num_msic, scale)

    def forward(self, X, Y, Z):
        Yi = self.MD(X)
        Yi = self.afun1(Yi)
        Zi = self.MR(X)
        Zi = self.afun2(Zi)
        dY = Y - Yi
        dZ = Z - Zi
        Xout = self.fusion(dY, dZ)
        return X + Xout


class Gradient_block(nn.Module):
    def __init__(self, num_hsic, num_msic):
        super(Gradient_block, self).__init__()
        self.conv_h = nn.Sequential(
            Get_gradient(num_hsic),
            nn.Conv2d(2 * num_hsic, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_m = nn.Sequential(
            Get_gradient(num_msic),
            nn.Conv2d(2 * num_msic, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, hsi, msi):
        X = self.conv_h(hsi)
        Y = self.conv_m(msi)
        return X, Y


class DSSFNet_with_grad(nn.Module):
    def __init__(self, num_hsic, num_msic, scale, I_num):
        self.I_num = I_num
        super(DSSFNet_with_grad, self).__init__()
        self.loop = nn.ModuleList()
        for i in range(I_num):
            self.loop.append(IRU(num_hsic, num_msic, scale))
        self.conv = nn.Conv2d(num_hsic * (I_num + 1), num_hsic, kernel_size=3, stride=1, padding=1)
        self.fusion = fusionunit(num_hsic, num_msic, scale)

    def forward(self, Y, Z):
        Xt = self.fusion(Y, Z)
        XDF = Xt
        for layers in self.loop[::]:
            Xt = layers(Xt, Y, Z)
            XDF = torch.cat((XDF, Xt), 1)
        X = self.conv(XDF)
        return X

class Discriminator(nn.Module):
    def __init__(self, nhsi):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nhsi, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
 
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
 
    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
