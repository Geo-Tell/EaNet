export PATH=/home/user/.conda/envs/py35torch1_0/bin:$PATH
CUDA_VISIBLE_DEVICES=0,1,4,5 python -m torch.distributed.launch --nproc_per_node=4 train.py
