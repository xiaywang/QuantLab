import torch
from torch import nn
import numpy as np


def update_mask(weights, mask, frac):
    assert frac >= 0.0 and frac <=1.0, "Invalid quantization fraction passed to inq.update_mask!"
    assert weights.shape == mask.shape, "Weights and mask_data passed to inq.update_mask have incompatible shape!"
    if frac == 0.0:
        mask.data = np.ones_like(mask.data)
        return
    if frac == 1.0:
        mask.data = np.zeros_like(mask.data)
        return
    data = weights[mask==0]
    #how much is already quantized?
    prev_quant = np.prod(list(mask[mask.data==0].size()))/np.prod(list(mask.size()))
    eff_quant_frac = (1-frac)/(1-prev_quant)
    dataSorted, _ = data.clone().contiguous().view(-1).abs_().cpu().sort()
    partition = int(len(dataSorted) * (1-frac)) - 1
    threshold = dataSorted[partition].item()
    return np.logical_or(mask, weights.abs()<threshold)
