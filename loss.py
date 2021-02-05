#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels
            invalid_mask = labels_cpu==self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min]<self.thresh else sorteds[self.n_min]
            labels[picks>thresh] = self.ignore_lb
        ## TODO: here see if torch or numpy is faster
        labels = labels.clone()
        loss = self.criteria(logits, labels)
        return loss

class ECELoss(nn.Module):
    def __init__(self, thresh, n_min, n_classes=19, alpha=1, radius=1, beta=0.5, ignore_lb=255, mode='ohem', *args, **kwargs):
        super(ECELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.n_classes = n_classes
        self.alpha = alpha
        self.radius = radius
        self.beta = beta

        if mode == 'ohem':
            self.criteria = OhemCELoss(thresh, n_min, ignore_lb=ignore_lb)
        elif mode == 'ce':
            self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb)
        else:
            raise Exception('No %s loss, plase choose form ohem and ce' % mode)

        self.edge_criteria = EdgeLoss(self.n_classes, self.alpha, self.radius)


    def forward(self, logits, labels):
        if self.beta > 0:
            return self.criteria(logits, labels) + self.beta*self.edge_criteria(logits, labels)
        else:
            return self.criteria(logits, labels)

class EdgeLoss(nn.Module):
    def __init__(self, n_classes=19, radius=1, alpha=1):
        super(EdgeLoss, self).__init__()
        self.n_classes = n_classes
        self.radius = radius
        self.alpha = alpha


    def forward(self, logits, label):
        prediction = F.softmax(logits, dim=1)
        ks = 2 * self.radius + 1
        filt1 = torch.ones(1, 1, ks, ks)
        filt1[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8
        filt1.requires_grad = False
        filt1 = filt1.cuda()
        label = label.unsqueeze(1)
        lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=self.radius)
        lbedge = 1 - torch.eq(lbedge, 0).float()

        filt2 = torch.ones(self.n_classes, 1, ks, ks)
        filt2[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8
        filt2.requires_grad = False
        filt2 = filt2.cuda()
        prededge = F.conv2d(prediction.float(), filt2, bias=None,
                            stride=1, padding=self.radius, groups=self.n_classes)

        norm = torch.sum(torch.pow(prededge,2), 1).unsqueeze(1)
        prededge = norm/(norm + self.alpha)


        # mask = lbedge.float()
        # num_positive = torch.sum((mask==1).float()).float()
        # num_negative = torch.sum((mask==0).float()).float()

        # mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        # mask[mask == 0] = 1.5 * num_positive / (num_positive + num_negative)

        # cost = torch.nn.functional.binary_cross_entropy(
            # prededge.float(),lbedge.float(), weight=mask, reduce=False)
        # return torch.mean(cost)
        return BinaryDiceLoss()(prededge.float(),lbedge.float())


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        return loss.sum()



if __name__ == '__main__':
    #criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    #criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria1 = ECELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = ECELoss(thresh=0.7, n_min=16*20*20//16).cuda()

    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, 10, 10] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    loss.backward()
    print('Done')
