import torch
from . import _C


def torch_image_dither(image):
    # image: gary image -> torch.Tensor with shape [H, W], range is [0, 1]
    sampled_xy = _C.image_dither(image)
    return sampled_xy
