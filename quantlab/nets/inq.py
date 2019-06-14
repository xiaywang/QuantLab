import torch
from torch import nn
import numpy as np


def update_mask(weights, mask, frac):
    assert frac >= 0.0 and frac <=1.0, "Invalid quantization fraction passed to inq.update_mask!"
    assert weights.shape == mask.shape, "Weights and mask_data passed to inq.update_mask have incompatible shape!"
    if frac == 0.0:
        mask.data = torch.ones_like(mask.data)
        return
    if frac == 1.0:
        mask.data = torch.zeros_like(mask.data)
        return
    #select unquantized weights
    data = weights[mask==1.0]
    data_len = np.prod(list(data.size()))
    #if we get a fully frozen weight set, allow 'thawing' 
    if data_len == 0:
        data = weights
        prev_quant = 0.0
    else:
        #how much is already quantized?
        prev_quant = np.prod(list(mask[mask.data==0].size()))/np.prod(list(mask.size()))
        eff_quant_frac = (frac-prev_quant)/(1-prev_quant)

    dataSorted, _ = data.clone().contiguous().view(-1).abs_().cpu().sort()
    partition = np.maximum(int(len(dataSorted) * (1-eff_quant_frac)) - 1, 0)
    if partition == 0:
        threshold = -1
    else:
        threshold = dataSorted[partition].item()
    return np.logical_and(mask, weights.abs()>threshold).float().to(weights.device)
