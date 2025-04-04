# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch.nn as nn
from models.swin_unet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=1):
        super(SwinUnet, self).__init__()
        self.swin_unet = SwinTransformerSys(
                            img_size        = img_size,
                            patch_size      = 4,
                            in_chans        = 3,
                            num_classes     = num_classes,
                            embed_dim       = 96,
                            depths          = [2, 2, 6, 2],
                            num_heads       = [3, 6, 12, 24],
                            window_size     = 4,
                            mlp_ratio       = 4.,
                            qkv_bias        = True,
                            qk_scale        = None,
                            drop_rate       = 0.0,
                            drop_path_rate  = 0.1,
                            ape             = False,
                            patch_norm      = True,
                            use_checkpoint  = False
                        )
                
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits
