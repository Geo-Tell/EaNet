#!/usr/bin/python
# -*- encoding: utf-8 -*-


class Config(object):
    def __init__(self):
        ## model and loss
        self.ignore_label = 255
        self.aspp_global_feature = True
        self.alpha = 0.01
        self.radius = 1
        self.beta = 0.5
        self.mode = 'ohem'
        ## dataset
        self.n_classes = 19
        self.datapth = '/home/hlx/MyResearch/Data/Segmentation'
        self.n_workers = 12
        self.crop_size = (768, 768)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        ## optimizer
        self.warmup_steps = 1000
        self.warmup_start_lr = 5e-8
        self.lr_start = 1e-4
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.lr_power = 0.9
        #self.max_iter = 41000
        self.max_iter = 41000
        ## training control
        self.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
        self.flip = True
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.ims_per_gpu = 3
        self.msg_iter = 100
        self.save_iter = 2000
        self.ohem_thresh = 0.7
        self.respth = './res101elkpp_cutmix_c'
        self.port = 32558
        self.resume = '/home/hlx/MyResearch/Segmentation/ELKPPNet/Redoing/res101elkpp_cutmix1/model_final_205.pth'
        ## eval control
        self.eval_batchsize = 2
        self.eval_n_workers = 4
        self.eval_scales = (1.0,) # (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0): 80.8323, (0.5, 0.75, 1.0, 1.25, 1.5, 1.75): 80.9061)
        self.eval_flip = False
        self.view_path = './res101elkpp_cutmix/view'

