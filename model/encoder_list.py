import torch
import torch.nn as nn
import numbers
from einops import rearrange


class UpSample_2(nn.Module):
    def __init__(self, in_channels):
        super(UpSample_2, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class UpSample_4(nn.Module):
    def __init__(self, in_channels):
        super(UpSample_4, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class UpSample_8(nn.Module):
    def __init__(self, in_channels):
        super(UpSample_8, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class channel_down(nn.Module):
    def __init__(self, in_channels):
        super(channel_down, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels, int(in_channels/2), 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, int(in_channels/2), 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class DownSample_same(nn.Module):
    def __init__(self, in_channels):
        super(DownSample_same, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample_same(nn.Module):
    def __init__(self, in_channels):
        super(UpSample_same, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels, int(in_channels/2), 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                               groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=bias)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# "FAM" corresponds to the "Adaptive Selective Intrinsic Supervised Feature Module (ASISF)" in the paper
class ASISF(nn.Module):
    def __init__(self, n_feat, n_res):
        super(ASISF, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_res, n_feat, kernel_size=3, padding=1)

    def forward(self, x, res):
        x1 = self.conv1(x)
        x1 = torch.sigmoid(x1)
        x2 = self.conv3(res)
        x1 = x1*x2
        return x1

##---------- Spatial Attention ----------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):

        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


#  Comprehensive Feature Attention (CFA)
class CFA(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(CFA, self).__init__()
        modules_body = [nn.Conv2d(n_feat, n_feat, kernel_size, padding=1), act, nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)]
        self.body = nn.Sequential(*modules_body)
        ## Pixel Attention
        self.SA = spatial_attn_layer()
        ## Channel Attention
        self.CA = CALayer(n_feat)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res

## "BasicBlock" corresponds to the "Bifocal Intrinsic-Context Attention (BICA)" in the paper
class BICA(nn.Module):
    def __init__(self, chns):
        super(BICA, self).__init__()

        self.norm = LayerNorm(chns, 'BiasFree')
        self.act = nn.GELU()

        self.DA = CFA(chns)

        self.Conv1 = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=1)
        self.Conv4 = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=1)
        self.Conv5 = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=1)
        self.Conv6 = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=1)

        self.conv_dialted = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=3, dilation=3)

        self.down = DownSample(chns)

        self.up = UpSample(chns*2)
        self.Conv1_1 = nn.Conv2d(chns*2, chns*2, kernel_size=3, stride=1, padding=1)
        self.Conv1_2 = nn.Conv2d(chns*2, chns*2, kernel_size=3, stride=1, padding=1)
        self.upsame = UpSample_same(chns*2)

    def forward(self, x):
        x = self.norm(x)

        x2 = self.DA(x)

        # Resolution-Guided Intrinsic Attention Module (ReGIA)
        x1 = x2
        x1 = self.down(x1)
        x1 = self.act(self.Conv1_1(x1))
        x1 = self.Conv1_2(x1)
        x1 = self.up(x1)
        x1 = torch.sigmoid(x1)
        x11 = x1 * x2

        # Hierarchical Context-Aware Feature Extraction (HCAFE)
        x3 = self.conv_dialted(x2)
        x4 = self.Conv4(x2)
        x22 = self.Conv2(x3 + x4)
        x22 = self.Conv5(self.act(x22))

        out = torch.cat([x11, x22], dim=1)
        out = self.upsame(out) + x2

        out = self.Conv6(self.act(out))

        return out


## Bifocal Intrinsic-Context Attention (BICA)
class BICA256(nn.Module):
    def __init__(self, chns):
        super(BICA256, self).__init__()

        self.norm = LayerNorm(chns, 'BiasFree')
        self.act = nn.GELU()
        self.DA = CFA(chns)

        self.Conv1 = SeparableConv2d(chns, chns, kernel_size=3)
        self.Conv2 = SeparableConv2d(chns, chns, kernel_size=3)
        self.Conv4 = SeparableConv2d(chns, chns, kernel_size=3)
        self.Conv5 = SeparableConv2d(chns, chns, kernel_size=3)
        self.Conv6 = SeparableConv2d(chns, chns, kernel_size=3)

        self.conv_dialted = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=3, dilation=3)

        self.down = DownSample(chns)
        self.up = UpSample(chns * 2)

        self.Conv1_1 = SeparableConv2d(chns * 2, chns * 2, kernel_size=3)
        self.Conv1_2 = SeparableConv2d(chns * 2, chns * 2, kernel_size=3)
        self.upsame = UpSample_same(chns*2)

    def forward(self, x):
        x = self.norm(x)
        x2 = self.DA(x)
        x1 = x2

        # Resolution-Guided Intrinsic Attention Module (ReGIA)
        x1 = self.down(x1)
        x1 = self.act(self.Conv1_1(x1))
        x1 = self.Conv1_2(x1)
        x1 = self.up(x1)
        x1 = torch.sigmoid(x1)
        x11 = x1 * x2


        # Hierarchical Context-Aware Feature Extraction (HCAFE)
        x3 = self.conv_dialted(x2)
        x4 = self.Conv4(x2)
        x22 = self.Conv2(x3 + x4)
        x22 = self.Conv5(self.act(x22))


        out = torch.cat([x11, x22], dim=1)
        out = self.upsame(out) + x2

        out = self.Conv6(self.act(out))
        return out

#########Stage4
class Encoder4(nn.Module):
    def __init__(self):
        super(Encoder4, self).__init__()

        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            BICA(32)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.block32 = BICA256(32)

        self.up2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.block32(x)
        res = x
        x = self.up2(x)
        return x,res


#########Stage3
class Encoder3(nn.Module):
    def __init__(self):
        super(Encoder3, self).__init__()

        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            BICA(32)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()

        self.layer21 = nn.Sequential(
            BICA(32)
        )

        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layer21(x)
        res = x
        x = self.layer24(x)

        return x,res

#########Stage3
class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()

        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        self.layer2 = nn.Sequential(
            BICA(32)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)

        return x


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()

        self.layer21 = BICA256(32)

        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.layer21(x)

        res = x

        x = self.layer24(x)

        return x,res


# Stage1
class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        # Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            BICA(32)
        )

        # Conv2
        self.layer5 = DownSample(32)
        self.layer6 = nn.Sequential(
            BICA(64)
        )

        # Conv3
        self.layer9 = DownSample(64)
        self.layer10 = nn.Sequential(
            BICA256(128)
        )

        self.layer12 = DownSample(128)
        self.layer13 = nn.Sequential(
            BICA256(256)
        )


        self.FAM_stage2 = ASISF(n_feat=64, n_res=32)
        self.FAM_stage3 = ASISF(n_feat=128, n_res=32)
        self.FAM_stage4 = ASISF(n_feat=256, n_res=32)


    def forward(self,x, stage2, stage3, stage4):

        x = self.layer1(x)
        x = self.layer2(x)

        res1 = x


        x = self.layer5(x)

        x1 = self.FAM_stage2(x, stage2)
        x = x + x1

        x = self.layer6(x)

        res2 = x

        x = self.layer9(x)

        x3 = self.FAM_stage3(x,stage3)
        x = x3 + x
        x = self.layer10(x)

        res3 = x
        x = self.layer12(x)

        x = self.FAM_stage4(x,stage4) + x
        x = self.layer13(x)

        return x,res3,res2,res1


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        # Deconv3
        self.layer12_1 = nn.Sequential(
            BICA256(256)
        )
        self.layer12_2 = UpSample(256)
        self.layer13 = nn.Sequential(
            BICA256(128)
        )

        self.layer16 = UpSample(128)

        self.layer17 = nn.Sequential(
            BICA(64)
        )

        self.layer20 = UpSample(64)

        self.layer21 = nn.Sequential(
            BICA(32)
        )

        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.FAM_stage2 = ASISF(n_feat=64, n_res=32)
        self.FAM_stage3 = ASISF(n_feat=128, n_res=32)
        self.FAM_stage4 = ASISF(n_feat=256, n_res=32)

        self.FAM_res3 = ASISF(n_feat=128, n_res=128)
        self.FAM_res2 = ASISF(n_feat=64, n_res=64)
        self.FAM_res1 = ASISF(n_feat=32, n_res=32)

    def forward(self, x, res3, res2, res1, res4_sam, res3_sam, res2_sam):

        x = self.layer12_1(x)
        x = self.FAM_stage4(x, res4_sam) + x
        x = self.layer12_2(x)

        x1 = self.FAM_stage3(x, res3_sam)
        x2 = self.FAM_res3(x, res3)
        x = x1 + x2 + x

        x = self.layer13(x)
        x = self.layer16(x)

        x1 = self.FAM_stage2(x, res2_sam)
        x2 = self.FAM_res2(x, res2)
        x = x + x1 + x2
        x = self.layer17(x)
        x = self.layer20(x)

        x = self.FAM_res1(x,res1) + x
        x = self.layer21(x)
        x = self.layer24(x)
        return x


