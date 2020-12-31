import torch


def generate_spixel(logits, img_in):
    """
    generate superpixels (Not used)

    Args:
        logits: torch.Tensor
            A Tensor of shape (b, nSpixel, h, w)
            that could be regarded as a soft assignment matrix
        img_in: torch.Tensor
            A Tensor of shape (b, c, h*w)
            whose prototype is normalized rgbxy image in default

    Return:
        avg_spixel_feat: tensor.Tensor
            A Tensor of shape (b, c, nSpixel)
    """
    # prob is a row-normalized (b, h*w, nSpixel) Tensor
    prob = logits.view(*logits.shape[:2], -1).permute(0, 2, 1)
    # avoid calculating invalid spixels
    spixel_cardinality = prob.sum(1, keepdim=True)
    spixel_mask = spixel_cardinality > 1e-5
    avg_spixel_feat = torch.bmm(img_in, prob) / (spixel_cardinality + 1e-5)
    avg_spixel_feat *= spixel_mask.float()

    return avg_spixel_feat


def get_hard_feat_map(logits, spixel_feat):
    """
    calculate a feature map after hard assignment (Not used)

    Args:
        logits: torch.Tensor
            A Tensor of shape (b, nSpixel, h, w)
            that could be regarded as a soft assignment matrix
        spixel_feat: torch.Tensor
            A Tensor of shape (b, c, nSpixel)

    Returns:
        H: torch.Tensor
            A Tensor of shape (b, c, h*w)
            where each c-dim vector is the feature of pixel's clustering center
        label: torch.Tensor
            A Tensor of shape (b, h, w)
            that indicates #spixel of each pixel
    """
    b, nSpixel, h, w = logits.shape
    c = spixel_feat.shape[1]
    label = logits.argmax(1).view(b, -1).long()
    b_idx = torch.arange(b)[:, None].repeat(1, h * w)
    H = spixel_feat[b_idx, :, label].squeeze(-2)
    H = H.permute(0, 2, 1).contiguous()

    return H, label.view(b, h, w)


def get_spixel_prob(spixel_x, spixel_y):
    """
        generate soft assignment matrix via probability vector in x and y direction

        Args:
            spixel_x: torch.tensor
                A trainable tensor indicating the probability that pixels belong
                to superpxiels in x direction
            spixel_y: torch.tensor
                A trainable tensor indicating the probability that pixels belong
                to superpxiels in y direction

        Return:
            spixel_prob: torch.tensor
                A trainable tensor indicating the probability that pixels belong
                to superpixels
    """
    b, _, h, w = spixel_x.shape
    return torch.einsum("bxhw,byhw->bxyhw", spixel_x,
                        spixel_y).view(b, -1, h, w).contiguous()
