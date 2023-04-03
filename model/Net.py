import torch
from torch import nn
from torch.nn import functional as F
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    # feat_mean = feat.view(N, C, -1).mean(dim=2)
    return feat_mean, feat_std


def normalization(X):
    size = X.size()
    X_mean, X_std, X_var = calc_mean_std(X)
    normalized_feat_X = (X - X_mean.expand(
        size)) / X_std.expand(size)
    return normalized_feat_X


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class SegHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        super(SegHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(inter_channels, channels, 1)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv_out(out)
        return out


class ResConvModule(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResConvModule, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = nn.Conv2d(in_c, out_c, 1, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResDeConvModule(nn.Module):
    def __init__(self, inc1, inc2=0):
        super(ResDeConvModule, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        if inc2 != 0:
            self.up = nn.ConvTranspose2d(inc2, inc2, kernel_size=2, padding=0, stride=2)
        else:
            self.up = nn.ConvTranspose2d(inc1, inc1, kernel_size=2, padding=0, stride=2)
        out_channel = (inc1 + inc2) // 4
        self.conv1 = nn.Conv2d(inc1 + inc2, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x = self.up(x2)
            x = torch.cat([x1, x], dim=1)
        else:
            x = self.up(x1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        identity = x

        x = self.conv2(x)
        x = self.bn2(x)

        out = x + identity

        out = self.relu(out)
        return out


class CatModule(nn.Module):
    def __init__(self, inc):
        super(CatModule, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inc, inc, 3, 1, 1),
                                  nn.BatchNorm2d(inc),
                                  nn.ReLU(inplace=True))
        self.bn = nn.BatchNorm2d(inc)
        self.conv_A = nn.Conv2d(inc, inc, 3, 1, 1)
        self.bn_A = nn.BatchNorm2d(inc)
        self.conv_B = nn.Conv2d(inc, inc, 3, 1, 1)
        self.bn_B = nn.BatchNorm2d(inc)
        self.conv_out = nn.Conv2d(inc * 2, inc * 2, 3, 1, 1)
        self.bn_out = nn.BatchNorm2d(inc * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, A, B):
        x_sub = self.conv(abs(A - B))
        return out


class FreBlock(nn.Module):
    def __init__(self, inc):
        super(FreBlock, self).__init__()
        self.convf = nn.Sequential(
            nn.Conv2d(inc * 2, inc * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc * 2, inc * 2, 1, 1, 0))
        self.convs = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inc, inc, 3, 1, 1))

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm='backward')
        x = self.convs(x)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = self.convf(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        return y + x


class apFreBlock(nn.Module):
    def __init__(self, inc, mode='phase'):
        super(apFreBlock, self).__init__()
        self.mode = mode
        self.conv_amp = nn.Sequential(
            nn.Conv2d(inc, inc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc, inc, 1, 1, 0))
        self.conv_pha = nn.Sequential(
            nn.Conv2d(inc, inc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc, inc, 1, 1, 0))
        self.conv = nn.Sequential(
            nn.Conv2d(inc * 2, inc * 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc * 2, inc * 2, 3, 1, 1))

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm='backward')
        amp = torch.abs(y)
        phase = torch.angle(y)
        amp = self.conv_amp(amp)
        phase = self.conv_pha(phase)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        # f = torch.cat([real, imag], dim=1)
        # f = self.conv(f)
        # real, imag = torch.chunk(f, 2, dim=1)
        y = torch.complex(real, imag)
        y = torch.abs(torch.fft.irfft2(y, s=(H, W), norm='backward'))
        return y


class StyleNorBlock(nn.Module):
    def __init__(self, inc):
        super(StyleNorBlock, self).__init__()
        self.conv_A = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc, inc, 3, 1, 1))
        self.conv_B = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc, inc, 3, 1, 1))
        self.conv1 = nn.Conv2d(inc, inc, 3, 1, 1)
        self.conv2 = nn.Conv2d(inc, inc, 3, 1, 1)
        self.conv3 = nn.Conv2d(inc, inc, 3, 1, 1)
        self.conv4 = nn.Conv2d(inc, inc, 3, 1, 1)
        self.conv5 = nn.Conv2d(inc, inc, 3, 1, 1)
        self.conv6 = nn.Conv2d(inc, inc, 3, 1, 1)

    def forward(self, A, B):
        # A = self.conv_A(A)
        # B = self.conv_B(B)
        A = adaptive_instance_normalization(A, B)
        A = self.conv1(A)
        B = adaptive_instance_normalization(B, A)
        B = self.conv2(B)

        A = adaptive_instance_normalization(A, B)
        A = self.conv3(A)
        B = adaptive_instance_normalization(B, A)
        B = self.conv4(B)

        A = adaptive_instance_normalization(A, B)
        A = self.conv5(A)
        B = adaptive_instance_normalization(B, A)
        B = self.conv6(B)

        # A = self.conv2(A)
        # B = self.conv2(B)
        # A = self.conv1(A)
        # B = self.conv2(B)
        return A, B


class ChannelExchange(nn.Module):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super(ChannelExchange, self).__init__()
        assert 0 <= p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


class SpatialExchange(nn.Module):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super(SpatialExchange, self).__init__()
        assert 0 <= p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2


class Adaptive_Filter_based_Exchange_Module(nn.Module):  # low-pass filter
    def __init__(self, inc):
        super(Adaptive_Filter_based_Exchange_Module, self).__init__()
        self.conv_cat = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc, inc // 2, 1, 1, 0))
        # self.insnor = nn.InstanceNorm2d(inc // 2)

    def forward(self, A, B):
        N, C, H, W = A.size()
        A = normalization(A)
        B = normalization(B)
        cat = torch.cat([A, B], dim=1)
        cat = self.conv_cat(cat)
        mean, std, var = calc_mean_std(cat)
        mean = mean.expand_as(cat)
        var = var.view(N, C, 1, 1).expand_as(cat)
        # chanel and spatial filter
        mask = cat < mean - var
        out_A, out_B = torch.zeros_like(A), torch.zeros_like(B)
        out_A[~mask] = A[~mask]
        out_B[~mask] = B[~mask]
        out_A[mask] = B[mask]
        out_B[mask] = A[mask]
        out_A = normalization(out_A)
        out_B = normalization(out_B)
        return out_A, out_B


class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(2 * channels, 2 * channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(2 * channels, 2 * channels, 1, 1, 0))
        self.post = nn.Conv2d(2 * channels, 2 * channels, 1, 1, 0)

    def forward(self, msf, panf):
        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf) + 1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf) + 1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp, panF_amp], 1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha, panF_pha], 1))

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)


class CD_Net(nn.Module):
    def __init__(self, ):
        super(CD_Net, self).__init__()

        self.ResConv1 = ResConvModule(3, 32)
        self.ResConv2 = ResConvModule(32, 64)
        self.ResConv3 = ResConvModule(64, 128)

        self.ResDeConv1 = ResDeConvModule(128, 128)
        self.ResDeConv2 = ResDeConvModule(128, 64)
        self.ResDeConv3 = ResDeConvModule(48)
        # self.deconv1 = DecBlock(256)
        # self.deconv2 = DecBlock(384)

        # self.AFEM1 = Adaptive_Filter_based_Exchange_Module(128)
        # self.AFEM2 = Adaptive_Filter_based_Exchange_Module(256)
        self.apFreBlock1 = apFreBlock(32)
        self.apFreBlock2 = apFreBlock(64)
        self.apFreBlock3 = apFreBlock(128)
        #
        self.FreBlock1 = FreBlock(32)
        self.FreBlock2 = FreBlock(64)
        self.FreBlock3 = FreBlock(128)
        # self.FreBlock4 = FreBlock(256)

        self.Freprocess1 = Freprocess(64)
        self.Freprocess2 = Freprocess(64)
        self.Freprocess3 = Freprocess(64)

        self.STBlock1 = StyleNorBlock(32)
        self.STBlock2 = StyleNorBlock(64)
        self.STBlock3 = StyleNorBlock(128)
        #
        # self.CatModule1 = CatModule(64)
        # self.CatModule2 = CatModule(128)
        #
        self.spex = SpatialExchange()
        self.chex = ChannelExchange()

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv_cat1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, ))

        self.conv_sub = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))

        self.conv_cat = nn.Conv2d(128, 128, 3, 1, 1, )
        # self.conv_cat2 = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, 1, ),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(256, 256, 3, 1, 1, ))
        # self.conv_cat3 = nn.Sequential(
        #     nn.Conv2d(512, 512, 3, 1, 1, ),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(512, 512, 3, 1, 1, ))

        # self.conv_cat1 = nn.Conv2d(128, 64, 1, 1, 0, )
        # self.conv_cat2 = nn.Conv2d(256, 128, 1, 1, 0, )
        # self.conv_cat3 = nn.Conv2d(512, 256, 1, 1, 0, )

        self.relu = nn.ReLU(inplace=True)
        self.SegHead3 = SegHead(64, 1)
        self.SegHead2 = SegHead(48, 1)
        self.SegHead1 = SegHead(12, 1)

    def forward(self, A, B, lbl=None):
        b, c, h, w = A.shape

        A1 = self.ResConv1(A)
        B1 = self.ResConv1(B)
        # A1, B1, cat1 = self.ACEM1(A1, B1)
        # A1 = self.apFreBlock1(A1)
        # B1 = self.apFreBlock1(B1)
        # A1 = self.FreBlock1(A1)
        # B1 = self.FreBlock1(B1)
        # A1, B1 = self.STBlock1(A1, B1)

        A2 = self.ResConv2(A1)
        B2 = self.ResConv2(B1)
        # A2, B2 = self.spex(A2, B2)
        # A2, B2 = self.chex(A2, B2)

        # A2 = self.FreBlock2(A2)
        # B2 = self.FreBlock2(B2)
        # A2, B2 = self.STBlock2(A2, B2)
        # A2 = self.apFreBlock2(A2)
        # B2 = self.apFreBlock2(B2)
        A3 = self.ResConv3(A2)
        B3 = self.ResConv3(B2)
        # A3, B3 = self.chex(A3, B3)
        # A3 = self.apFreBlock3(A3)
        # B3 = self.apFreBlock3(B3)
        # A3 = self.FreBlock3(A3)
        # B3 = self.FreBlock3(B3)
        # A3, B3 = self.STBlock3(A3, B3)
        A1 = self.conv1(A1)
        B1 = self.conv1(B1)
        A2 = self.conv2(A2)
        B2 = self.conv2(B2)
        A3 = self.conv3(A3)
        B3 = self.conv3(B3)

        # A2 = self.FreBlock2(A2)
        # B2 = self.FreBlock2(B2)

        cat1_ = self.conv_sub(abs(A1 - B1))
        cat2_ = self.conv_sub(abs(A2 - B2))
        cat3_ = self.conv_sub(abs(A3 - B3))

        # cat1 = self.Freprocess3(A1, B1)
        # cat2 = self.Freprocess3(A2, B2)
        # cat3 = self.Freprocess3(A3, B3)
        cat1 = torch.cat([A1, B1], dim=1)
        cat1 = self.conv_cat1(cat1 + cat1_)

        # cat1 = self.CatModule1(A1, B1)

        # cat1 = self.FreBlock3(cat1)
        # A2, B2 = self.STBlock3(A2, B2)

        cat2 = torch.cat([A2, B2], dim=1)  # mul-scale change detection module
        cat2 = self.conv_cat1(cat2 + cat2_)

        cat3 = torch.cat([A3, B3], dim=1)
        cat3 = self.conv_cat1(cat3 + cat3_)

        # cat2 = self.CatModule2(A2, B2)
        # cat2 = self.FreBlock4(cat2)

        cat2 = self.ResDeConv1(cat2, cat3)
        # cat2 = self.deconv1(cat2)
        # cat = torch.cat([cat1, cat2], dim=1)

        cat1 = self.ResDeConv2(cat1, cat2)
        cat = self.ResDeConv3(cat1)
        # cat = self.deconv2(cat)

        result3 = torch.sigmoid(F.interpolate(self.SegHead3(cat2), size=[h, w], mode='bicubic', align_corners=True))
        result2 = torch.sigmoid(F.interpolate(self.SegHead2(cat1), size=[h, w], mode='bicubic', align_corners=True))
        result1 = torch.sigmoid(self.SegHead1(cat))
        # result = F.interpolate(result, size=[h, w], mode='bicubic', align_corners=True)
        # if not self.training:
        return result1, result2, result3
        # loss_cd = hybrid_loss(result, lbl)
        # loss_cd = dice_loss(result, lbl)
        # return loss_cd
