import torch.nn as nn
import torch
import torchvision.models as models
import math
import torch.nn.functional as F

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)  # (window_size, 1)
    _2D_window = _1D_window @ _1D_window.t()  # (window_size, window_size)
    _2D_window = _2D_window.float().unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True):
    (_, channel, _, _) = img1.size()
    if window is None:
        window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.window = None
        self.channel = 1

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if self.window is None or channel != self.channel:
            self.window = create_window(self.window_size, channel).to(img1.device)
            self.channel = channel
        # SSIM 越高表示越相似，因此作为损失我们用 1 - SSIM
        ssim_val = ssim(img1, img2, window=self.window, window_size=self.window_size, size_average=self.size_average)
        return 1 - ssim_val

class TotalGenLoss(nn.Module):
    def __init__(self, is_cuda):
        super(TotalGenLoss, self).__init__()
        self.vgg = VGGContent()
        self.ssim_loss = SSIMLoss(window_size=11, size_average=True)
        if is_cuda:
            self.vgg = self.vgg.cuda()
            self.ssim_loss = self.ssim_loss.cuda()

    def forward(self, org_image, gen_image):
        vgg_org_image = self.vgg(org_image)
        vgg_gen_image = self.vgg(gen_image)
        bs = org_image.size(0)

        content_loss = ((vgg_org_image - vgg_gen_image) ** 2).mean(1)
        mse_gen_loss = (torch.abs(org_image - gen_image)).view(bs, -1).mean(1)
        ssim_loss_val = self.ssim_loss(org_image, gen_image)
        total_loss = (0.7 * mse_gen_loss + 0.3 * content_loss).mean() + 0.1 * ssim_loss_val
        return total_loss


class VGGContent(nn.Module):
    def __init__(self):
        super(VGGContent, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True).features

    def forward(self, x):
        bs = x.size(0)
        return self.vgg(x).view(bs, -1)
    


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduction_ratio = min(reduction_ratio, channels)
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super().__init__()
        reduced_dim = max(channels // reduction_ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_weights = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * channel_weights

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(spatial))
        return x * spatial_weights

def build_conv_block(in_chans, out_chans, kernel_size=3, stride=2, padding=1, use_bn=True, bn_momentum=0.8, use_leaky=False):
    layers = []
    layers.append(nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding))
    if use_leaky:
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    else:
        layers.append(nn.ReLU(inplace=True))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_chans, momentum=bn_momentum))
    return nn.Sequential(*layers)


def build_deconv_block(in_chans, out_chans, use_bn=True):
    layers = []
    layers.append(nn.Upsample(scale_factor=2,
                              mode="bilinear", align_corners=True))
    layers.append(nn.Conv2d(in_chans, out_chans, 3, 1, 1))
    layers.append(nn.ReLU(inplace=True))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_chans, momentum=0.8))
    return nn.Sequential(*layers)


def build_conv_block2(in_chans, out_chans, kernel_size=3, stride=2, padding=1, 
                    use_bn=True, bn_momentum=0.8, use_leaky=False):
    layers = [
        nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding)
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_chans, momentum=bn_momentum))
    layers.append(nn.LeakyReLU(0.2, inplace=True) if use_leaky else nn.ReLU(inplace=True))
    layers.append(CBAM(out_chans))  # CBAM在ReLU后
    return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels*2, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels*3, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1 = self.lrelu(self.conv1(x))
        out2 = self.lrelu(self.conv2(torch.cat([x, out1], dim=1)))
        out3 = self.lrelu(self.conv3(torch.cat([x, out1, out2], dim=1)))
        return out3 * 0.2 + x  # 残差缩放
    
class RRDB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x
    
class SuperResolutionModule(nn.Module):
    def __init__(self, channels=64, num_blocks=3):
        super().__init__()
        self.blocks = nn.Sequential(*[RRDB(channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = self.blocks(x)
        x = self.conv(x)
        return x + residual

class CBAMGenerator(nn.Module):
    def __init__(self, n_feats=32):
        super().__init__()
        # 下采样卷积（顺序：Conv -> BN -> ReLU -> CBAM）
        self.conv1 = build_conv_block2(3, n_feats, 5, padding=2, use_bn=False, use_leaky=True)
        self.conv2 = build_conv_block2(n_feats, n_feats*4, 4, use_leaky=True)
        self.conv3 = build_conv_block2(n_feats*4, n_feats*8, 4, use_leaky=True)
        self.conv4 = build_conv_block2(n_feats*8, n_feats*8, use_leaky=True)
        self.conv5 = build_conv_block2(n_feats*8, n_feats*8, use_leaky=True)

        # 额外卷积层（手动添加CBAM，顺序：Conv -> ReLU -> CBAM）
        self.add_conv1 = nn.Conv2d(n_feats*8, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cbam_add1 = CBAM(64)
        self.add_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.cbam_add2 = CBAM(64)
        self.add_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.cbam_add3 = CBAM(64)

        # 残差块（不修改）
        self.res_block1 = ResidualBlock()
        self.res_block2 = ResidualBlock()
        self.res_block3 = ResidualBlock()
        self.res_block4 = ResidualBlock()
        self.res_block5 = ResidualBlock()

        # 反卷积块（顺序：Upsample -> Conv -> BN -> ReLU -> CBAM）
        self.deconv1 = self._deconv_block2(n_feats*2, n_feats*8)
        self.deconv2 = self._deconv_block2(n_feats*(8+8), n_feats*8)
        self.deconv3 = self._deconv_block2(n_feats*(8+8), n_feats*4)
        self.deconv4 = self._deconv_block2(n_feats*(4+4), n_feats*1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.sr_module = SuperResolutionModule(channels=n_feats*2)

        self.final = nn.Conv2d(n_feats*2, 3, 3, padding=1)
        self.cbam_final = CBAM(3, reduction_ratio=1)  # 强制 reduction_ratio=1
        self.act = nn.Tanh()

    def _deconv_block2(self, in_chans, out_chans, use_bn=True):
        layers = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_chans, out_chans, 3, padding=1)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_chans, momentum=0.8))
        layers.append(nn.ReLU(inplace=True))
        layers.append(CBAM(out_chans))  # CBAM在ReLU后
        return nn.Sequential(*layers)

    def forward(self, x):
        # 下采样
        d1 = self.conv1(x)    # Conv -> ReLU -> CBAM
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)

        # 额外卷积层（顺序：Conv -> ReLU -> CBAM）
        a1 = self.cbam_add1(self.relu(self.add_conv1(d5)))
        a2 = self.cbam_add2(self.relu(self.add_conv2(a1)))
        bridge = self.cbam_add3(self.relu(self.add_conv3(a2)))

        # 残差块（无CBAM）
        bridge = self.res_block1(bridge)
        bridge = self.res_block2(bridge)
        bridge = self.res_block3(bridge)
        bridge = self.res_block4(bridge)
        bridge = self.res_block5(bridge)
        bridge += a1

        # 上采样
        u1 = torch.cat([self.deconv1(bridge), d4], dim=1)
        u2 = torch.cat([self.deconv2(u1), d3], dim=1)
        u3 = torch.cat([self.deconv3(u2), d2], dim=1)
        u4 = torch.cat([self.deconv4(u3), d1], dim=1)
        u4 = self.up(u4)

        sr_out = self.sr_module(u4)
        
        # 最终输出
        final = self.final(sr_out)
        final = self.cbam_final(final)  # Conv -> CBAM
        return self.act(final)


class FUnIEDiscriminator(nn.Module):
    def __init__(self, n_feats=32):
        super(FUnIEDiscriminator, self).__init__()

        # Build discriminator blocks
        self.block1 = self._block(3*2, n_feats, False)
        self.block2 = self._block(n_feats, n_feats*2)
        self.block3 = self._block(n_feats*2, n_feats*4)
        self.block4 = self._block(n_feats*4, n_feats*8)

        # Validility block
        # In this work, kernel size is 3 instead of 4
        self.validility = nn.Conv2d(n_feats*8, 1, 3, 1, 1)

    def _block(self, in_chans, out_chans, use_bn=True):
        layers = []
        layers.append(nn.Conv2d(in_chans, out_chans, 3, 2, 1))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_chans, momentum=0.8))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # (B, 6, 256, 256)
        x = self.block1(x)              # (B, 32, 128, 128)
        x = self.block2(x)              # (B, 64, 64, 64)
        x = self.block3(x)              # (B, 128, 32, 32)
        x = self.block4(x)              # (B, 256, 16, 16)
        valid = self.validility(x)      # (B, 1, 16, 16)
        return valid.squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, n_feats=64):
        super(ResidualBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_feats, momentum=0.8))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_feats, momentum=0.8))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        x = self.block(x)
        return x + identity


    def forward(self, x):
        # Downsample
        d1 = self.conv1(x)   # (B, 32, 128, 128)
        d2 = self.conv2(d1)  # (B, 128, 64, 64)
        d3 = self.conv3(d2)  # (B, 256, 32, 32)
        d4 = self.conv4(d3)  # (B, 256, 16, 16)
        d5 = self.conv5(d4)  # (B, 256, 8, 8)

        # Additional conv layers
        a1 = self.relu(self.add_conv1(d5))  # (B, 64, 8, 8)
        a2 = self.relu(self.add_conv2(a1))
        bridge = self.relu(self.add_conv3(a2))

        # Residual blocks
        bridge = self.res_block1(bridge)
        bridge = self.res_block2(bridge)
        bridge = self.res_block3(bridge)
        bridge = self.res_block4(bridge)
        bridge = self.res_block5(bridge)
        bridge += a1

        # Upsample
        u1 = torch.cat([self.deconv1(bridge), d4], dim=1)  # (B, 512, 16, 16)
        u2 = torch.cat([self.deconv2(u1), d3], dim=1)      # (B, 512, 32, 32)
        u3 = torch.cat([self.deconv3(u2), d2], dim=1)      # (B, 256, 64, 64)
        u4 = torch.cat([self.deconv4(u3), d1], dim=1)      # (B, 64, 128, 128)
        u4 = self.up(u4)                                   # (B, 64, 256, 256)
        return self.act(self.final(u4))



if __name__ == "__main__":
    model = FUnIEDiscriminator()
    x1 = torch.rand(1, 3, 256, 256)
    x2 = torch.rand(1, 3, 256, 256)
    print(model(x1, x2).size())

    model = VGGContent()
    x = torch.rand(1, 3, 256, 256)
    print(model(x).size())

