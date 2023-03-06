import torch
from torchvision.models import resnet18, resnet50
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torch.nn import functional as F
from utils.losses import hybrid_loss, dice_loss


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def normalization(A, B):
    assert (A.size()[:2] == B.size()[:2])
    size = A.size()
    A_mean, A_std = calc_mean_std(A)
    B_mean, B_std = calc_mean_std(B)
    normalized_feat_A = (A - A_mean.expand(
        size)) / A_std.expand(size)
    normalized_feat_B = (B - B_mean.expand(
        size)) / B_std.expand(size)
    return normalized_feat_A, normalized_feat_B


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
        self.relu = nn.ReLU()

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


class SiamConvModule(nn.Module):
    def __init__(self, in_c, out_c):
        super(SiamConvModule, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 2, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, A, B):
        A = self.conv1(A)
        B = self.conv1(B)
        A = self.conv2(A)
        B = self.conv2(B)
        A = self.relu(A)
        B = self.relu(B)
        return A, B


class FreBlock(nn.Module):
    def __init__(self, inc):
        super(FreBlock, self).__init__()
        self.convf = nn.Sequential(
            nn.Conv2d(inc * 2, inc * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc * 2, inc * 2, 1, 1, 0))
        self.convs = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
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
        # y = torch.complex(real, imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        # y = torch.fft.rfft2(y, norm='backward')
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
        A, B = normalization(A, B)
        A = self.conv1(A)
        B = self.conv2(B)
        # B = normalization(B, A)
        # A = self.conv2(A)
        # B = self.conv2(B)
        return A, B


class CD_Net(nn.Module):
    def __init__(self, ):
        super(CD_Net, self).__init__()
        self.conv1 = SiamConvModule(3, 64)
        self.conv2 = SiamConvModule(64, 128)

        self.resconv1 = ResConvModule(3, 64)
        self.resconv2 = ResConvModule(64, 128)

        self.FreBlock1 = FreBlock(64)
        self.FreBlock2 = FreBlock(128)

        self.apFreBlock1 = FreBlock1(64)
        self.apFreBlock2 = FreBlock1(128)

        self.STBlock1 = StyleNorBlock(64)
        self.STBlock2 = StyleNorBlock(128)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1))
        self.relu = nn.ReLU()

        self.SegHead = SegHead(256, 2)

    def forward(self, A, B, lbl=None):
        b, c, h, w = A.shape
        # A, B = self.conv1(A, B)
        A = self.resconv1(A)
        B = self.resconv1(B)
        A, B = self.STBlock1(A, B)
        # A = self.FreBlock1(A)
        # B = self.FreBlock1(B)
        A = self.apFreBlock1(A)
        B = self.apFreBlock1(B)

        # A, B = self.conv2(A, B)
        A = self.resconv2(A)
        B = self.resconv2(B)
        A, B = self.STBlock2(A, B)
        # A = self.FreBlock2(A)
        # B = self.FreBlock2(B)
        A = self.apFreBlock2(A)
        B = self.apFreBlock2(B)

        cat = torch.cat([A, B], dim=1)
        cat = self.conv_cat(cat)

        pred = self.SegHead(cat)
        result = F.interpolate(pred, size=[h, w], mode='bilinear', align_corners=False)
        if not self.training:
            return result
        loss_cd = dice_loss(result, lbl)
        return loss_cd
