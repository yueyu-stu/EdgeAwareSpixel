import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16

import copy
import math

from .losses import Loss
from .model_utils import get_spixel_prob


def conv_in(in_c, out_c, kernel=3, isPad=True, isRelu=True):
    layers = []
    if isPad:
        layers.append(nn.ReflectionPad2d(1))
    layers.append(nn.Conv2d(in_c, out_c, kernel, bias=False))
    layers.append(nn.InstanceNorm2d(out_c, affine=True))
    if isRelu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SpixelCNN(nn.Module):
    """
        This work is inspired by T. Suzuki. Superpixel Segmentation via
        Convolutional Neural Networks with Regularized Information Maximization,
        In ICASSP, 2020.
        Source code of ICASSP'20 could be found at
        https://github.com/DensoITLab/ss-with-RIM

        Differences:
        1) Contour-aware mechanism is added in SmoothnessLoss. This loss turns
        to SmoothContourLoss.
        2) spixel_out layer (with n_spixel channels) turns to 2 layers that
           generate superpixels in x and y directions, respectively.
           torch.enisum() is employed to generate a global probability map.
    """

    def __init__(self,
                 img_shape,
                 num_spixels=100,
                 in_c=3,
                 num_feat=32,
                 num_layers=4,
                 device="cuda"):
        super(SpixelCNN, self).__init__()

        # extract features
        layers = []
        for idx in range(num_layers):
            layers.append(conv_in(in_c, num_feat << idx))
            in_c = num_feat << idx
        self.feat_conv = nn.Sequential(*layers)

        # reconstruct image
        self.recon_conv = conv_in(in_c, 3, 1, False, False)

        # generate assignment matrix
        num_spixels_w = int(
            math.sqrt(num_spixels * img_shape[1] / img_shape[0]))
        num_spixels_h = int(
            math.sqrt(num_spixels * img_shape[0] / img_shape[1]))
        self.spixel_conv_x = conv_in(in_c, num_spixels_w, 1, False, False)
        self.spixel_conv_y = conv_in(in_c, num_spixels_h, 1, False, False)

    def forward(self, x):
        feat = self.feat_conv(torch.cat([x[:, :3, ...], x[:, -2:, ...]], 1))

        # generate reconstructed images
        recon_img = self.recon_conv(feat)

        # obtain soft assignment matrix
        spixel_x = self.spixel_conv_x(feat)
        spixel_y = self.spixel_conv_y(feat)
        spixel_prob = get_spixel_prob(spixel_x, spixel_y)

        return spixel_prob.softmax(1), recon_img

    # weights and biases initialization
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def optimize(self,
                 img,
                 img_in,
                 num_iter=500,
                 lr=1e-2,
                 loss_weights=[1, 2, 10],
                 sc_weights=[1, 1],
                 thresh=1.0,
                 coef_card=2,
                 sigma=2,
                 margin=1,
                 device="cuda"):
        optimizer = torch.optim.Adam(self.parameters(), lr)
        loss = Loss(loss_weights, sc_weights, thresh, coef_card, sigma, margin)

        for idx in range(num_iter):

            spixel_prob, recon_img = self.forward(img_in)
            total_loss, losses = loss(spixel_prob, recon_img, img_in)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"[{idx+1}/{num_iter}] total_loss: {total_loss.item(): .6f},\
                mutual_loss: {losses[0].item(): .6f}, \
                sc_loss: {losses[1].item(): .6f},\
                recon_loss: {losses[2]: .6f}, ")
