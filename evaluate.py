#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import *
from models.elkpp_whole import ELKPPNet
from cityscapes import CityScapes
from configs_eval import config_factory
from view_utils import colormap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import sys
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import numba
import argparse
#from scipy.misc import imsave
import scipy
from PIL import Image

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    return parse.parse_args()


class MscEval(object):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.distributed = dist.is_initialized()
        ## dataloader
        dsval = CityScapes(cfg, mode='val')
        sampler = None
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dsval)
        self.dl = DataLoader(dsval,
                        batch_size = cfg.eval_batchsize,
                        sampler = sampler,
                        shuffle = False,
                        num_workers = cfg.eval_n_workers,
                        drop_last = False)


    def __call__(self, net, mode=None):
        ## evaluate
        hist_size = (self.cfg.n_classes, self.cfg.n_classes)
        hist = np.zeros(hist_size, dtype=np.float32)
        if dist.is_initialized() and dist.get_rank()!=0:
            diter = enumerate(self.dl)
        else:
            diter = enumerate(tqdm(self.dl))
        for i, (imgs, label, impth) in diter:
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.cfg.n_classes, H, W))
            probs.requires_grad = False
            for sc in self.cfg.eval_scales:
                new_hw = [int(H*sc), int(W*sc)]
                with torch.no_grad():
                    im = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
                    im = im.cuda()
                    out = net(im)
                    out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
                    prob = F.softmax(out, 1)
                    probs += prob.cpu()
                    if self.cfg.eval_flip:
                        out = net(torch.flip(im, dims=(3,)))
                        out = torch.flip(out, dims=(3,))
                        out = F.interpolate(out, (H, W), mode='bilinear',
                                align_corners=True)
                        prob = F.softmax(out, 1)
                        probs += prob.cpu()
                    del out, prob
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)
            
            # visualization
            if mode == 'view':
                if not os.path.isdir(self.cfg.view_path):
                    os.makedirs(self.cfg.view_path)
                for k in range(N):
                    color_pred = Image.fromarray(colormap(np.squeeze(preds[k]), self.cfg.n_classes).astype('uint8'))
                    color_lb = Image.fromarray(colormap(np.squeeze(label.data[k].numpy()), self.cfg.n_classes).astype('uint8'))
                    color_pred.save(os.path.join(self.cfg.view_path, 'pred_{}.png'.format(impth[k])))
                    color_lb.save(os.path.join(self.cfg.view_path, 'gt_{}.png'.format(impth[k])))
            
            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        if self.distributed:
            hist = torch.tensor(hist).cuda()
            dist.all_reduce(hist, dist.ReduceOp.SUM)
            hist = hist.cpu().numpy().astype(np.float32)
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU

    @numba.jit
    def compute_hist(self, pred, lb):
        n_classes = self.cfg.n_classes
        keep = np.logical_not(lb==self.cfg.ignore_label)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist


def evaluate():
    ## setup
    cfg = config_factory['resnet_cityscapes']
    args = parse_args()
    if not args.local_rank == -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
                    backend = 'nccl',
                    init_method = 'tcp://127.0.0.1:{}'.format(cfg.port),
                    world_size = torch.cuda.device_count(),
                    rank = args.local_rank
                    )
        setup_logger(cfg.respth)
    else:
        FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
        log_level = logging.INFO
        if dist.is_initialized() and dist.get_rank()!=0:
            log_level = logging.ERROR
        logging.basicConfig(level=log_level, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = ELKPPNet(cfg)
    save_pth = '/home/hlx/MyResearch/Segmentation/ELKPPNet/Redoing/res101elkpp_cutmix1/model_final_205.pth'
    net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.eval()
    if not args.local_rank == -1:
        net = nn.parallel.DistributedDataParallel(net,
                device_ids = [args.local_rank, ],
                output_device = args.local_rank
                )

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(cfg)
    mIOU = evaluator(net, mode=None)
    logger.info('mIOU is: {:.6f}'.format(mIOU))


if __name__ == "__main__":
    evaluate()
