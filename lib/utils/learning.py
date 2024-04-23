import os
import numpy as np
import torch
import torch.nn as nn
from functools import partial

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_backbone(args):
    print(f'Backbone type: {args.backbone}')
    if args.backbone == 'SiC_dynamicTUP':
        from lib.model.SiC_dynamicTUP import Skeleton_in_Context as SiC_dynamicTUP
        model_backbone = SiC_dynamicTUP(args, dim_in=args.dim_in, dim_out=args.dim_out, dim_feat=args.dim_feat, dim_rep=args.dim_rep,
                                           depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                           maxlen=args.maxlen, num_joints=args.data.num_joints)
    
    elif args.backbone == 'SiC_staticTUP':
        from lib.model.SiC_staticTUP import Skeleton_in_Context as SiC_staticTUP
        model_backbone = SiC_staticTUP(args, dim_in=args.dim_in, dim_out=args.dim_out, dim_feat=args.dim_feat, dim_rep=args.dim_rep,
                                           depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                           maxlen=args.maxlen, num_joints=args.data.num_joints)
    else:
        raise Exception("Undefined backbone type.")
    return model_backbone