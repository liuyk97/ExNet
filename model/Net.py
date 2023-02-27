import torch
from torchvision.models import resnet18, resnet50
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torch.nn import functional as F
from utils.losses import hybrid_loss, dice_loss

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

vgg.load_state_dict(torch.load('model/vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:31])


def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def deep_fourier_trans(A, B):
    _, _, H, W = A.shape
    A_freq = torch.fft.rfft2(A, norm='backward')
    B_freq = torch.fft.rfft2(B, norm='backward')
    A_amp = torch.abs(A_freq)
    B_amp = torch.abs(B_freq)
    A_phase = torch.angle(A_freq)
    B_phase = torch.angle(B_freq)
    real = B_amp * torch.cos(A_phase)
    imag = B_amp * torch.sin(A_phase)
    x_recom = torch.complex(real, imag) + 1e-8
    x_recom = torch.fft.irfft2(x_recom, s=(H, W), norm='backward') + 1e-8
    x_recom = torch.abs(x_recom) + 1e-8
    return x_recom


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


class Style_Net(nn.Module):
    def __init__(self, g_vis=False, pred=False):
        super(Style_Net, self).__init__()
        self.g_vis = g_vis
        self.pred = pred
        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu = nn.ReLU()
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.dice_loss = dice_loss
        self.SegHead = SegHead(1024, 2)
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)

    def forward(self, content, style, cd_lbl, alpha=1.0):
        assert 0 <= alpha <= 1
        input_shape = content.shape[-2:]
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adaptive_instance_normalization(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g = self.decoder(t)

        g_feats = self.encode_with_intermediate(g)
        g_cd_feats = self.conv2(self.conv1(g_feats[-1]))
        style_cd_feats = self.conv2(self.conv1(style_feats[-1]))
        pred = self.SegHead(torch.cat([g_cd_feats, style_cd_feats[-1]], dim=1))
        result = F.interpolate(pred, size=input_shape, mode='bilinear', align_corners=False)
        if not self.training:
            return result
        loss_cd = self.dice_loss(result, cd_lbl)
        loss_c = self.calc_content_loss(g_feats[-1], content_feat)
        loss_s = self.calc_style_loss(g_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_feats[i], style_feats[i])
        return loss_c, loss_s, loss_cd


class SiamConvModule(nn.Module):
    def __init__(self, in_c, out_c, ins_nor=False):
        super(SiamConvModule, self).__init__()
        self.ins_nor = ins_nor
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 2, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, A, B):
        A = self.conv1(A)
        B = self.conv1(B)
        A = self.conv2(A)
        B = self.conv2(B)
        if self.ins_nor:
            A = adaptive_instance_normalization(A, B)
        A = self.relu(A)
        B = self.relu(B)
        return A, B


class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm='backward')
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = self.conv1(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        return y


class CD_Net(nn.Module):
    def __init__(self, ):
        super(CD_Net, self).__init__()
        # self.resnet18 = resnet18()
        self.ins_nor = False
        self.fourier_trans = False
        self.conv1 = SiamConvModule(3, 64)
        self.conv2 = SiamConvModule(64, 128)
        self.FreBlock1 = FreBlock(256)
        self.FreBlock2 = FreBlock(256)
        self.relu = nn.ReLU()

        # self.conv3 = SiamConvModule(128, 256, self.ins_nor, self.fourier_trans)
        # self.FreBlock = FreBlock(256)
        self.SegHead = SegHead(256, 2)
        self.mse_loss = nn.MSELoss()

    def forward(self, A, B, lbl=None):
        input_shape = A.shape[-2:]
        A, B = self.conv1(A, B)
        A, B = self.conv2(A, B)
        A = self.FreBlock1(A)
        B = self.FreBlock1(B)
        A = self.FreBlock2(A)
        B = self.FreBlock2(B)
        # A, B = self.conv2(A, B)
        # A, B = self.conv3(A, B)
        # if self.fourier_trans:
        #     A = self.FreBlock(A, B)
        # A, B = self.conv4(A, B)

        pred = self.SegHead(torch.cat([A, B], dim=1))
        result = F.interpolate(pred, size=input_shape, mode='bilinear', align_corners=False)
        if not self.training:
            return result
        loss_cd = dice_loss(result, lbl)
        # loss_c = self.mse_loss(A, A_t)
        loss_c = torch.tensor(0)
        return loss_cd, loss_c
