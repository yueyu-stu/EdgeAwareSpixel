import torch
from torch import nn


class MutualInfoLoss(nn.Module):
    """
        Mutual Information Loss derived from ss-with-RIM that also applied in
        this work.
        First term enforces to generate a sparse nSpixel dimension vector for
        each pixel; Second term indicates the cardinality of each spixel.

        Args:
            logits: torch.tensor
                A trainable tensor of shape (b, nSpixel, h, w) that
                represents the probability of each pixel belonging to all spixels.
                It should be softmaxed before calling this loss funtion.

            coef: float
                A coefficient that controls the amplitude of second term.
    """

    def __init__(self, coef=2):
        super().__init__()
        self.coef = coef

    def forward(self, logits):
        pixel_wise_ent = -(logits * torch.log(logits + 1e-16)).sum(1).mean()
        marginal_prob = logits.mean((2, 3))
        marginal_ent = -(marginal_prob *
                         torch.log(marginal_prob + 1e-16)).sum(1).mean()

        return pixel_wise_ent - self.coef * marginal_ent


class SmoothContourLoss(nn.Module):
    """
        Loss function that contains smoothness loss derived from ss-with-RIM
        and contour-aware loss.
        Smoothness loss concerns about smoothness of local patterns, while
        contour-aware loss is interested in whether two patches are divided.
        Cross entropy (or KL divergence) is applied to calculate Contour-aware 
        loss. When calculating the gradients of probability, i.e. dp, and that 
        of image, i.e. di, we desire that the distribution of dp should appoximate
        that of di.

        Args:
            logits: torch.tensor
                A trainable tensor of shape (b, nSpixel, h, w)
                It should be softmaxed before calling this loss function.
            image: torch.tensor
                A tensor derived from color channels of input with shape
                (b, c, h, w)
            sigma: float
                Parameter of transformed Gaussian kernel size
            weights: List[float]
                A List contains 2 coefficients that control the amplitudes of
                2 losses
            thresh: float
                Parameter for controling the amplitude of edge
            margin: int
                Parameter for finding edge width
    """

    def __init__(self, sigma=2, weights=[1, 1], thresh=1.0, margin=1):
        super().__init__()
        self.sigma = 2 * sigma**2
        self.weights = weights
        self.thresh = thresh
        self.margin = margin

    def forward(self, logits, image):
        dp, di = self.get_gradients(logits, image)

        smooth = 0.
        contour = 0.
        for idx in range(len(dp)):
            smooth += self.smooth_loss(dp[idx], di[idx])
            contour += self.contour_loss(dp[idx], di[idx])

        return self.weights[0] * smooth + self.weights[1] * contour

    def get_gradients(self, logits, image):
        dp_dx = (logits[..., :-self.margin] - logits[..., self.margin:])
        dp_dy = (logits[..., :-self.margin, :] - logits[..., self.margin:, :])
        di_dx = (image[..., :-self.margin] - image[..., self.margin:])
        di_dy = (image[..., :-self.margin, :] - image[..., self.margin:, :])

        return [dp_dx, dp_dy], [di_dx, di_dy]

    def smooth_loss(self, dp, di):
        return (dp.abs().sum(1) *
                (-di.pow(2).sum(1) / self.sigma).exp()).mean()

    def contour_loss(self, dp, di):
        di_norm = di.pow(2)
        di_min = (di_norm.min(-1,
                              keepdim=True).values).min(-2,
                                                        keepdim=True).values
        di_max = (di_norm.max(-1,
                              keepdim=True).values).max(-2,
                                                        keepdim=True).values
        di_norm = ((di_norm - di_min) / (di_max - di_min + 1e-16)).sum(1) * 2
        isValidEdges = di_norm > self.thresh

        dp_valid = dp.abs().sum(1) * isValidEdges
        di_valid = di_norm * isValidEdges

        return -(di_valid * (torch.log(dp_valid + 1e-16))).mean()


class ReconstructionLoss(nn.Module):
    """
        Reconstruction loss for validating whether the features extracted by
        CNN are effective

    Args:
        pred_img: torch.tensor
            A trainable tensor of shape (b, c, h, w), where c = 3 in default
        image: torch.tensor
            A tensor of shape (b, c, h, w) that is derived from color channels
            of input
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_img, image):
        return self.mse_loss(pred_img, image)


class Loss(nn.Module):
    """
        Calculate total loss

        Args:
            pred_spixel: List[List[torch.tensor]]
                containing 2 Lists with spixel probabilities in x and y dir
            pred_img: tensor.tensor
                A trainable tensor of shape (b, c, h, w), where c is 3 in default
            img_in: tensor.tensor
                An input tensor of shape (b, c, h, w), where c is 20 in default
                These features consists of 15-channel features extracted by CNN,
                and 5-channel RGB ()
            loss_weights: List[float]
                Weights for MutualInfoLoss, SmoothContourLoss, CompactnessLoss
                and Reconstruction in sequence
            sc_weights: List[float]
                Weights for SmoothContourLoss
            thresh: float
                A parameter for SmoothContourLoss
            coef_card: float
                Parameter for MutualInfoLoss
            sigma: float
                Parameter for SmoothContourLoss
            margin: int
                Parameter for SmoothContourLoss

        Returns:
            total_loss: torch.tensor
                A trainable tensor indicating final loss
            losses: List[torch.tensor]
                A List containing aforementioned losses in sequence
    """

    def __init__(self,
                 loss_weights=[1, 2, 10],
                 sc_weights=[1, .75],
                 thresh=1.0,
                 coef_card=2,
                 sigma=2,
                 margin=1):
        super().__init__()
        # parameters - weights
        self.weights = loss_weights

        # Loss classes
        self.mutual_info_loss = MutualInfoLoss(coef_card)
        self.sc_loss = SmoothContourLoss(sigma, sc_weights, thresh, margin)
        self.recon_loss = ReconstructionLoss()

    def forward(self, spixel_prob, recon_img, img_feat):
        loss_mutual = self.mutual_info_loss(spixel_prob)
        # use blurred & clahed image
        loss_sc = self.sc_loss(spixel_prob, img_feat[:, 3:-2, ...])
        # use original image
        loss_recon = self.recon_loss(recon_img, img_feat[:, :3, ...])
        loss_total = 0.

        losses = [loss_mutual, loss_sc, loss_recon]

        for idx in range(len(losses)):
            loss_total += self.weights[idx] * losses[idx]

        return loss_total, losses
