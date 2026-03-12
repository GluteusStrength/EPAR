# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
There are some modifications from the original code (I-JEPA: CVPR 2023)
"""

import math

from multiprocessing import Value

import torch

class MaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=8,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(1., 1.),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
        mode="train"
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size # patch size 8 -> 28, 28
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes
        self.mode = mode

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self,
                           generator,
                           scale,
                           row_range,
                           col_range,
                           aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item() # (0, 1) -> random float
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        height = row_range[1] - row_range[0] + 1
        width = col_range[1] - col_range[0] + 1
        max_keep = int(height*width*mask_scale)
        # max_keep = int(height*width*scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar) # at least min_ar, max: max_ar
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        h = max(h, 2)
        w = max(w, 2)
        while h >= height:
            if h < 3:
                break
            h -= 1
        while w >= width:
            if w < 3:
                break
            w -= 1
        return (h, w)

    def _sample_block_mask(self, 
                           b_size,
                           row_range,
                           col_range,
                           min_keep,
                           acceptable_regions=None):
        h, w = b_size # b_size is up to ...
        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions.to(mask.dtype)
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            t_min, t_max = row_range[0], row_range[1] + 1 - h
            c_min, c_max = col_range[0], col_range[1] + 1 - w
            if t_min == t_max:
                top = t_min
            else:
                top = torch.randint(t_min, t_max, (1,)) # row
            if c_min == c_max:
                left = c_min
            else:
                left = torch.randint(c_min, c_max, (1,)) # column
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None: # format: Mask Complement format: [28 x 28] shape -> patch size: 8
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            # min_keep_pred = int(min_keep*self.pred_mask_scale[0])
            min_keep_pred = self.min_keep
            valid_mask = len(mask) >= min_keep_pred
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, 
                 bsz,
                 row_range,
                 col_range,
                 target_acceptable_regions=None,
                 context_indices=None):
        '''
        For AD-JEPA, we followed the strategy of I-JEPA.
        There are some modifications
        # 1. Get the foreground maps which is the target_acceptable_regions.
        # 2. sample multiple target blocks
        # 3. Encoder blocks are the remaining regions
        '''
        B = bsz
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            row_range=row_range,
            col_range=col_range,
            aspect_ratio_scale=self.aspect_ratio)

        collated_masks_pred = []
        collated_masks_enc = []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        min_keep = 999
        for b in range(B):
            min_keep = min(torch.sum(target_acceptable_regions[b]), min_keep)
        for b in range(B):
            masks_p = []
            for i in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size,
                                                       row_range=row_range,
                                                       col_range=col_range,
                                                       acceptable_regions=target_acceptable_regions[b],
                                                       min_keep=min_keep)
                masks_p.append(mask)
                target_acceptable_regions[b] = target_acceptable_regions[b] * mask_C # not to overlap
                min_keep_pred = min(len(mask), min_keep_pred)
            collated_masks_pred.append(masks_p)
            
            masks_e = []
            # remaining areas are all context blocks
            for _ in range(1): # Single
                tgt_area = torch.cat([t.flatten() for t in collated_masks_pred[b]])
                tgt_values = tgt_area.unique()
                context_blocks = context_indices[~torch.isin(context_indices, tgt_values)]
                min_keep_enc = min(len(context_blocks), min_keep_enc)
                masks_e.append(context_blocks)
            collated_masks_enc.append(masks_e)
        
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        
        return collated_masks_enc, collated_masks_pred