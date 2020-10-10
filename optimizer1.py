#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import logging
import torch.distributed as dist
from ranger import Ranger


logger = logging.getLogger()

class Optimizer(object):
    def __init__(self,
                model,
                lr0,
                momentum,
                wd,
                warmup_steps,
                warmup_start_lr,
                max_iter,
                power,
                start_iter = 0):
        if hasattr(model, 'module'):
            back_wd_params, back_no_wd_params, wd_params, no_wd_params = model.module.get_params()
        else:
            back_wd_params, back_no_wd_params, wd_params, no_wd_params = model.get_params()
        params_list = [{'params': back_wd_params, 'lr': lr0, 'weight_decay': wd},
                {'params': back_no_wd_params, 'lr':  lr0, 'weight_decay': 0},
                {'params': wd_params, 'lr': lr0, 'weight_decay': wd},
                {'params': no_wd_params, 'lr': lr0, 'weight_decay': 0},]
        
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = start_iter
        #'''
        self.optim = torch.optim.SGD(
                params_list,
                lr = lr0,
                momentum = momentum,
                weight_decay = wd)
        '''

        self.optim = Ranger(
                params_list,
                lr = lr0,
                weight_decay = wd)
        '''
        self.warmup_factor = (self.lr0 / self.warmup_start_lr) ** (1. / self.warmup_steps)

    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr * (self.warmup_factor ** self.it)
        else:
            factor = (1 - (self.it - self.warmup_steps) / (self.max_iter - self.warmup_steps)) ** self.power
            lr = self.lr0 * factor + self.warmup_start_lr*0.01*(1-factor)
        return lr
    def step(self):
        self.lr = self.get_lr()

        for indx, pg in enumerate(self.optim.param_groups):
            pg['lr'] = self.lr
        self.optim.defaults['lr'] = self.lr
            
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2 and dist.get_rank()==0:
            logger.info('==> warmup done, start to implement poly lr strategy')
    '''
    def step(self):
        self.lr = self.get_lr()
        if self.it <= self.warmup_steps:
            for pg in self.optim.param_groups:
                pg['lr'] = self.lr
            self.optim.defaults['lr'] = self.lr
        else:
            for pg in self.optim.param_groups[:2]:
                pg['lr'] = 0.01*self.lr
            for pg in self.optim.param_groups[2:]:
                pg['lr'] = self.lr
            self.optim.defaults['lr'] = self.lr
            
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2 and dist.get_rank()==0:
            logger.info('==> warmup done, start to implement poly lr strategy')
    '''
    def zero_grad(self):
        self.optim.zero_grad()


