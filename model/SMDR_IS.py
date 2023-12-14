
import torch.nn.functional as F
import math
from model.encoder_list import *


## Adaptive Selective Intrinsic Supervised Feature Module (ASISF)
class ASISF(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(ASISF, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias)
        self.conv3 = nn.Conv2d(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, img):
        x1 = self.conv1(x)
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1


class UpSample_samec(nn.Module):
    def __init__(self, in_channels):
        super(UpSample_samec, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SMDR_IS(nn.Module):
    def __init__(self):
        super(SMDR_IS, self).__init__()

        self.encoder_1 = Encoder1()
        self.encoder_2 = Encoder2()
        self.encoder_3 = Encoder3()
        self.encoder_4 = Encoder4()
        self.decoder_1 = Decoder1()
        self.decoder_2 = Decoder2()
        self.decoder_3 = Decoder3()
        self.decoder_4 = Decoder4()

        self.sam4 = ASISF(n_feat=32, kernel_size=1, bias=False)
        self.sam3 = ASISF(n_feat=32, kernel_size=1, bias=False)
        self.sam2 = ASISF(n_feat=32, kernel_size=1, bias=False)


    def forward(self, x):
        B, C, H, W = x.size()
        if H%128!=0:
            h1 = math.ceil(H / 128)  * 128
            pad = nn.ReflectionPad2d(padding=(0, 0, 0, h1 - H))
            x = pad(x)
        if W%128!=0:
            w1 = math.ceil(W / 128)  * 128
            pad = nn.ReflectionPad2d(padding=(0, w1 - W, 0, 0))
            x = pad(x)
        x_2x = F.upsample(x, scale_factor=0.5)
        x_4x = F.upsample(x_2x, scale_factor=0.5)
        x_8x = F.upsample(x_4x, scale_factor=0.5)

        stage4 = self.encoder_4(x_8x)
        out_8,fea4 = self.decoder_4(stage4)

        res4_sam = self.sam4(fea4, out_8)


        stage3_input = x_4x
        stage3 = self.encoder_3(stage3_input)
        out_4, fea3 = self.decoder_3(stage3)

        res3_sam = self.sam3(fea3, out_4)


        stage2_input = x_2x
        stage2 = self.encoder_2(stage2_input)
        out_2,fea2 = self.decoder_2(stage2)

        res2_sam = self.sam2(fea2, out_2)

        stage1_input = x
        stage1, stage1_res3, stage1_res2, stage1_res1 = self.encoder_1(stage1_input,stage2,stage3,stage4)
        out = self.decoder_1(stage1, stage1_res3, stage1_res2, stage1_res1,res4_sam,res3_sam,res2_sam)

        return out,out_2,out_4,out_8


