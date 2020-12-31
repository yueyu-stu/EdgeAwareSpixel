import torch
import torch.nn.functional as F

import numpy as np

from skimage import io
from skimage.util import img_as_float
from skimage.exposure import equalize_adapthist
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from utils.swf import SideWindowBoxFilter


def preprocess(img_name, device="cuda"):
    img = io.imread(img_name)

    # 0, 1, 2 channel - original image
    image = img_as_float(img)
    image = torch.from_numpy(image).permute(2, 0, 1).float()[None]
    # 3, 4, 5 channel - filtered and clahed image
    blur = SideWindowBoxFilter(device=device).to(device)
    img_bc = blur.forward(img)
    img_bc = img_as_float(equalize_adapthist(img_bc))
    img_bc = torch.from_numpy(img_bc).permute(2, 0, 1).float()[None]
    # 6, 7 channels - coordinates
    h, w = image.shape[-2:]
    coord = torch.stack(torch.meshgrid(torch.arange(h),
                                       torch.arange(w))).float()[None]

    # instance normalization
    img_in = torch.cat([image, img_bc, coord], 1).to(device)
    img_in = (img_in - img_in.mean((2, 3), keepdim=True)) / \
        img_in.std((2, 3), keepdim=True)

    return img, img_in


def calc_spixel(spixel_labels, num_spixel, device="cuda"):
    spix = spixel_labels.squeeze().to("cpu").detach().numpy()

    segment_size = spix.size / num_spixel
    min_size = int(0.06 * segment_size)
    max_size = int(3.0 * segment_size)
    spix = _enforce_label_connectivity_cython(spix[None], min_size,
                                              max_size)[0]

    return spix
