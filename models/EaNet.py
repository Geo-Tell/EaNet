#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo
import torch.nn.functional as F
import torchvision

from .resnet import Resnet50, Resnet101
from modules import InPlaceABNSync as BatchNorm2d
#from torch.nn import BatchNorm2d
#from apex.parallel import SyncBatchNorm as BatchNorm2d


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = True)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class HADCLayer(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, dilation=1, mode='parallel', *args, **kwargs):
        super(HADCLayer, self).__init__()
        self.mode = mode
        self.ks = ks
        if ks > 3:
            padding = int(dilation*((ks-1)//2))
            if mode == 'cascade':
                self.hadc_layer = nn.Sequential(ConvBNReLU(in_chan, out_chan,
                                                  ks=[3, ks], dilation=[1, dilation],
                                                  padding=[1, padding]),
                                       ConvBNReLU(out_chan, out_chan,
                                                  ks=[ks, 3], dilation=[dilation, 1],
                                                  padding=[padding, 1]))
            elif mode == 'parallel':
                self.hadc_layer1 = ConvBNReLU(in_chan, out_chan,
                                              ks=[3, ks], dilation=[1, dilation],
                                              padding=[1, padding])
                self.hadc_layer2 = ConvBNReLU(in_chan, out_chan,
                                              ks=[ks, 3], dilation=[dilation, 1],
                                              padding=[padding, 1])
            else:
                raise Exception('No %s mode, please choose from cascade and parallel' % mode)

        elif ks ==3 :
            self.hadc_layer = ConvBNReLU(in_chan, out_chan, ks=ks, dilation=dilation, padding=dilation)

        else:
            self.hadc_layer = ConvBNReLU(in_chan, out_chan, ks=ks, dilation=1, padding=0)
        
        self.init_weight()

    def forward(self, x):
        if self.mode == 'cascade' or self.ks <= 3:
            return self.hadc_layer(x)
        elif self.mode == 'parallel' and self.ks > 3:
            x1 = self.hadc_layer1(x)
            x2 = self.hadc_layer2(x)
            return x1 + x2


    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class LKPBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ks, dilation=[1, 2, 3], mode='parallel', *args, **kwargs):
        super(LKPBlock, self).__init__()
        if ks >= 3:
            self.lkpblock = nn.Sequential(HADCLayer(in_chan, out_chan,
                                                    ks=ks, dilation=dilation[0], mode=mode),
                                  HADCLayer(out_chan, out_chan,
                                            ks=ks, dilation=dilation[1], mode=mode),
                                  HADCLayer(out_chan, out_chan,
                                          ks=ks, dilation=dilation[2], mode=mode))
        else:
            self.lkpblock = HADCLayer(in_chan, out_chan, ks=ks)
        
        self.init_weight()
        
    def forward(self, x):
        return self.lkpblock(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class LKPP(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, ks_list=[7, 5, 3, 1], mode='parallel', with_gp=True, *args, **kwargs):
        super(LKPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = LKPBlock(in_chan, out_chan, ks=ks_list[0], dilation=[1, 2, 3], mode=mode)
        self.conv2 = LKPBlock(in_chan, out_chan, ks=ks_list[1], dilation=[1, 2, 3], mode=mode)
        self.conv3 = LKPBlock(in_chan, out_chan, ks=ks_list[2], dilation=[1, 2, 3], mode=mode)
        self.conv4 = LKPBlock(in_chan, out_chan, ks=ks_list[3], mode=mode)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, ks=1)
            self.conv_out = ConvBNReLU(out_chan*5, out_chan, ks=1)
        else:
            self.conv_out = ConvBNReLU(out_chan*4, out_chan, ks=1)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
            
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
            
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    
    def get_params(self):
        wd_params = []
        non_wd_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'bias' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params


class Decoder(nn.Module):
    def __init__(self, n_classes, low_chan=[1024, 512, 256], *args, **kwargs):
        super(Decoder, self).__init__()
        self.conv_16 = ConvBNReLU(low_chan[0], 256, ks=3, padding=1)
        self.conv_8 = ConvBNReLU(low_chan[1], 128, ks=3, padding=1)
        self.conv_4 = ConvBNReLU(low_chan[2], 64, ks=3, padding=1)
        self.conv_fuse1 = ConvBNReLU(256, 128, ks=3, padding=1)
        self.conv_fuse2 = ConvBNReLU(128, 64, ks=3, padding=1)
        self.conv_fuse3 = ConvBNReLU(64, 64, ks=3, padding=1)
                
        
        self.fuse = ConvBNReLU(64, 64, ks=3, padding=1)

        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=1, bias=False)

        self.init_weight()

    def forward(self, feat4, feat8, feat16, feat_lkpp):
        H, W = feat16.size()[2:]
        feat16_low = self.conv_16(feat16)
        feat8_low = self.conv_8(feat8)
        feat4_low = self.conv_4(feat4)
        feat_lkpp_up = F.interpolate(feat_lkpp, (H, W), mode='bilinear',
                align_corners=True)
        
        feat_out = self.conv_fuse1(feat16_low+feat_lkpp_up)
        H, W = feat8_low.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                align_corners=True)
        feat_out = self.conv_fuse2(feat_out+feat8_low)
        H, W = feat4_low.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                align_corners=True)
        feat_out = self.conv_fuse3(feat_out+feat4_low)
        
        logits = self.conv_out(self.fuse(feat_out))
        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params = []
        non_wd_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'bias' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params

class EaNet(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(ELKPPNet, self).__init__()
        self.backbone = Resnet101(stride=16)
        self.lkpp = LKPP(in_chan=2048, out_chan=256, mode='parallel', with_gp=cfg.aspp_global_feature)
        self.decoder = Decoder(cfg.n_classes, low_chan=[1024, 512, 256])
        #  self.backbone = Darknet53(stride=16)
        #  self.aspp = ASPP(in_chan=1024, out_chan=256, with_gp=False)
        #  self.decoder = Decoder(cfg.n_classes, low_chan=128)

        #self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat4, feat8, feat16, feat32 = self.backbone(x)
        feat_lkpp = self.lkpp(feat32)
        logits = self.decoder(feat4, feat8, feat16, feat_lkpp)
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)

        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    '''
    def get_params(self):
        back_bn_params, back_no_bn_params = self.backbone.get_params()
        tune_wd_params = list(self.lkpp.parameters())  \
                + list(self.decoder.parameters())  \
                + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params
    '''
    def get_params(self):
        bk_wd_params, bk_no_wd_params = self.backbone.get_params()
        lkpp_wd_params, lkpp_no_wd_params = self.lkpp.get_params()
        decoder_wd_params, decoder_no_wd_params = self.decoder.get_params()
        wd_params = bk_wd_params  \
                + lkpp_wd_params  \
                + decoder_wd_params
        no_wd_params = bk_no_wd_params  \
                + lkpp_no_wd_params  \
                + decoder_no_wd_params
        
        return wd_params, no_wd_params




if __name__ == "__main__":
    from configurations import Config
    cfg = Config()
    net = ELKPPNet(cfg)
    net.cuda()
    net.train()
    #net = nn.DataParallel(net)
    for i in range(200):
        #  with torch.no_grad():
        in_ten = torch.randn((1, 3, 768, 768)).cuda()
        logits = net(in_ten)
        print(i)
        print(logits.size())
