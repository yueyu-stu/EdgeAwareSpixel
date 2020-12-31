import os

import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

import torch

from utils.config import *
from utils.data_process import preprocess, calc_spixel
from models.spixel_model import SpixelCNN

# initialization
device = "cuda" if torch.cuda.is_available() else "cpu"

# get dataset
img_list = os.listdir(IMG_PATH)
img_list.remove("Thumbs.db")

# get superpixels in inference process
for img_name in img_list:
    print(img_name)
    # image preprocessing
    img, img_in = preprocess(os.path.join(IMG_PATH, img_name))

    # get model
    model = SpixelCNN(num_spixels=NUM_SPIXELS,
                      img_shape=img.shape[:2],
                      in_c=IN_CHANNELS,
                      num_feat=NUM_FEAT,
                      num_layers=NUM_LAYERS,
                      device=device).to(device)

    # model weights initialization
    model.weight_init()

    # optimization
    model.optimize(img_in=img_in,
                   img=img,
                   num_iter=NUM_ITER,
                   lr=LR,
                   loss_weights=LOSS_WEIGHTS,
                   sc_weights=SC_WEIGHTS,
                   thresh=THRESH,
                   coef_card=COEF_CARD,
                   sigma=SIGMA,
                   margin=MARGIN,
                   device=device)

    # generate sparse assignment matrix spxiel_prob
    spixel_prob, _ = model.forward(img_in)

    # obtain superpixel labels, save segmented image and .npy file for measurement
    label = calc_spixel(spixel_prob.argmax(1).long(), NUM_SPIXELS)
    plt.imsave(
        os.path.join(OUT_IMG, str(NUM_SPIXELS), img_name[:-4] + "_bdry.png"),
        mark_boundaries(img, label))
    np.save(os.path.join(OUT_NPY, str(NUM_SPIXELS), img_name[:-4] + "_spixel"),
            label)
