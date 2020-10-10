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
import cv2
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
trainId2id = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33}

class MscEval(object):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.distributed = dist.is_initialized()
        ## dataloader
        dsval = CityScapes(cfg, mode='test')
        sampler = None
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dsval)
        self.dl = DataLoader(dsval,
                        batch_size = cfg.eval_batchsize,
                        sampler = sampler,
                        shuffle = False,
                        num_workers = cfg.eval_n_workers,
                        drop_last = False)


    def __call__(self, net):
        ## evaluate
        hist_size = (self.cfg.n_classes, self.cfg.n_classes)
        hist = np.zeros(hist_size, dtype=np.float32)
        if dist.is_initialized() and dist.get_rank()!=0:
            diter = enumerate(self.dl)
        else:
            diter = enumerate(tqdm(self.dl))
        for i, (imgs, path) in diter:
            N, _, H, W = imgs.shape
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
            if not os.path.isdir(self.cfg.view_path+'_multi'):
                os.makedirs(self.cfg.view_path+'_multi')
            color_pred = Image.fromarray(colormap(np.squeeze(preds), self.cfg.n_classes).astype('uint8'))
            color_pred.save(os.path.join(self.cfg.view_path+'_multi', '{}'.format(path[0])))

            # visualization
            # if not os.path.isdir(self.cfg.view_path+'_label'):
                # os.makedirs(self.cfg.view_path+'_label')
            # color_pred = np.squeeze(preds).astype('uint8')
            # for k,v in trainId2id.items():
                # color_pred[color_pred == k] = v
            # cv2.imwrite(os.path.join(self.cfg.view_path+'_label', '{}'.format(path[0])), color_pred)

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
    save_pth = '/home/hlx/MyResearch/Segmentation/ELKPPNet/Redoing/res101elkpp_cutmix_c/model_final_40.pth'
    net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.eval()
    if not args.local_rank == -1:
        net = nn.parallel.DistributedDataParallel(net,
                device_ids = [args.local_rank, ],
                output_device = args.local_rank
                )

    ## Visualization
    logger.info('Visualization')
    evaluator = MscEval(cfg)
    evaluator(net)


if __name__ == "__main__":
    evaluate()
