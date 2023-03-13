import torch
from torch import nn
from torch.nn import functional as F
from utils.losses import dice_loss


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def normalization(X):
    size = X.size()
    X_mean, X_std = calc_mean_std(X)
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
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecBlock(nn.Module):
    def __init__(self, inc):
        super(DecBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(inc, inc // 2, kernel_size=2, padding=0, stride=2)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inc, inc, 3, 1, 1))
        self.conv_out = nn.Conv2d(inc, inc, 3, 1, 1)
        self.bn = nn.BatchNorm2d(inc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x2 = self.deconv(x2)
        out = self.conv_cat(torch.cat([x1, x2], dim=1))
        out = self.relu(self.bn(self.conv_out(out)))
        # out = self.relu(normalization(self.conv_out(out)))
        return out


class DecBlock1(nn.Module):
    def __init__(self, inc):
        super(DecBlock1, self).__init__()
        self.deconv = nn.ConvTranspose2d(inc, inc, kernel_size=2, padding=0, stride=2)
        self.conv_out = nn.Conv2d(inc, inc, 3, 1, 1)
        self.bn = nn.BatchNorm2d(inc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        out = self.relu(self.bn(self.conv_out(x)))
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


class FreBlock1(nn.Module):
    def __init__(self, inc, mode='phase'):
        super(FreBlock1, self).__init__()
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
            nn.Conv2d(inc * 2, inc * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc * 2, inc * 2, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm='backward')
        amp = torch.abs(y)
        phase = torch.angle(y)
        amp = self.conv_amp(amp)
        phase = self.conv_pha(phase)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        f = torch.cat([real, imag], dim=1)
        f = self.conv(f)
        real, imag = torch.chunk(f, 2, dim=1)
        y = torch.complex(real, imag)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        return y + x


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

    def forward(self, A, B):
        A = self.conv_A(A)
        B = self.conv_B(B)
        A = normalization(A)
        B = normalization(B)
        A = self.conv1(A)
        B = self.conv2(B)
        return A, B


class CD_Net(nn.Module):
    def __init__(self, ):
        super(CD_Net, self).__init__()

        self.resconv1 = ResConvModule(3, 64)
        self.resconv2 = ResConvModule(64, 128)

        # self.deconv1 = DecBlock(128)
        self.deconv1 = DecBlock1(256)
        self.deconv2 = DecBlock1(384)
        self.deconv3 = DecBlock1(384)
        # self.conv_out = nn.Conv2d(128, 128, 3, 1, 0)
        # self.bn = nn.BatchNorm2d(64)

        self.FreBlock1 = FreBlock(64)
        self.FreBlock2 = FreBlock(128)
        self.FreBlock3 = FreBlock(128)
        self.FreBlock4 = FreBlock(256)

        self.apFreBlock1 = FreBlock1(64)
        self.apFreBlock2 = FreBlock1(128)
        self.apFreBlock3 = FreBlock1(128)

        self.STBlock1 = StyleNorBlock(64)
        self.STBlock2 = StyleNorBlock(128)
        self.STBlock3 = StyleNorBlock(128)

        self.conv_cat1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1))
        self.conv_cat2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.mse_loss = nn.MSELoss()

        self.SegHead = SegHead(384, 2)

    def forward(self, A, B, lbl=None):
        b, c, h, w = A.shape

        A1 = self.resconv1(A)
        B1 = self.resconv1(B)

        # A1 = self.FreBlock1(A1)
        # B1 = self.FreBlock1(B1)
        # A1 = self.apFreBlock1(A1)
        # B1 = self.apFreBlock1(B1)
        # A1, B1 = self.STBlock1(A1, B1)

        A2 = self.resconv2(A1)
        B2 = self.resconv2(B1)

        # A2 = self.FreBlock2(A2)
        # B2 = self.FreBlock2(B2)
        # A2 = self.apFreBlock2(A2)
        # B2 = self.apFreBlock2(B2)
        # A2, B2 = self.STBlock2(A2, B2)

        # A2 = self.deconv1(A1, A2)
        # B2 = self.deconv1(B1, B2)

        cat1 = torch.cat([A1, B1], dim=1)
        cat1 = self.conv_cat1(cat1)

        # A2 = self.FreBlock3(A2)
        # B2 = self.FreBlock3(B2)
        # A2 = self.apFreBlock3(A2)
        # B2 = self.apFreBlock3(B2)
        # A2, B2 = self.STBlock3(A2, B2)

        cat2 = torch.cat([A2, B2], dim=1)
        cat2 = self.conv_cat2(cat2)

        cat2 = self.deconv1(cat2)
        cat = torch.cat([cat1, cat2], dim=1)
        cat = self.deconv2(cat)

        cat = self.deconv3(cat)
        # cat = self.FreBlock4(cat)
        # cat = self.apFreBlock3(cat)
        result = self.SegHead(cat)
        # result = F.interpolate(result, size=[h, w], mode='bilinear', align_corners=True)
        # loss_s = self.calc_style_loss(A1, B1) + self.calc_style_loss(A2, B2)
        if not self.training:
            return result
        loss_cd = dice_loss(result, lbl)
        return loss_cd
