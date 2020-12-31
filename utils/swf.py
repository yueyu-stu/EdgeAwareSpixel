"""
    Side Window Filter
    H. Yin, Y. Gong, and G. Qiu. Side Window Filtering. In CVPR, 2019.
    Source code:
        https://github.com/YuanhaoGong/SideWindowFilter (Matlab ver.)
        https://github.com/wang-kangkang/SideWindowFilter-pytorch (Pytorch ver.)
    This is derived from the second link.
"""
import numpy as np
import torch
import torch.nn as nn


class SideWindowBoxFilter(nn.Module):
    def __init__(self, radius=1, iteration=5, device="cuda"):
        super(SideWindowBoxFilter, self).__init__()

        self.radius = radius
        self.iteration = iteration
        self.device = device

        r = radius
        k = np.ones((2 * r + 1, 1)) / (2 * r + 1)  # separable kernel
        k_L = k.copy()
        k_L[r + 1:] = 0
        k_L = k_L / np.sum(k_L)  # half kernel
        k_R = k_L[::-1, :]

        kernel1 = np.array([k_L, k_L, k_R, k_R, k_L, k_R, k,
                            k]).astype(np.float32)
        kernel2 = np.array(
            [k_L.T, k_R.T, k_L.T, k_R.T, k.T, k.T, k_L.T,
             k_R.T]).astype(np.float32)
        kernel1 = kernel1[:, np.newaxis, :, :]
        kernel2 = kernel2[:, np.newaxis, :, :]
        self.conv1 = nn.Conv2d(1,
                               8,
                               kernel_size=2 * r + 1,
                               bias=False,
                               padding=(self.radius, 0))
        self.conv1.weight = torch.nn.Parameter(torch.from_numpy(kernel1))
        self.conv2 = nn.Conv2d(8,
                               8,
                               kernel_size=2 * r + 1,
                               bias=False,
                               padding=(0, self.radius),
                               groups=8)
        self.conv2.weight = torch.nn.Parameter(torch.from_numpy(kernel2))

    def forward(self, img):
        # prapare img
        img = np.float32(img) / 255
        U = np.pad(img, ((self.radius, self.radius),
                         (self.radius, self.radius), (0, 0)), 'edge')
        for i in range(self.iteration):

            U_input = U
            U_input = U_input.transpose((2, 0, 1))
            U_input = U_input[:, np.newaxis, :, :]
            U_input = torch.from_numpy(U_input).to(self.device)

            # forward
            output1 = self.conv1(U_input)
            output2 = self.conv2(output1)
            d = output2 - U_input

            #abs and index
            d_abs = torch.abs(d)
            dm = torch.min(d_abs, dim=1, keepdim=True)
            dm_index = dm[1]
            dm = torch.gather(d, 1, dm_index)

            # get and return
            dm = dm.cpu().detach().numpy()
            dm = dm[:, 0, :, :]
            dm = dm.transpose((1, 2, 0))
            U = U + dm

        U = U[self.radius:-self.radius, self.radius:-self.radius, :] * 255
        return U.astype(np.uint8)
