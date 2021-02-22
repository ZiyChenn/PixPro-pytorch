import copy
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Wapper(nn.Module):
    def __init__(self,
                 backbone,
                 instance_layer='avgpool',
                 instance_shape=128,
                 pixel_layer='layer4',
                 pixel_shape=256):
        super(Wapper, self).__init__()

        self.backbone = backbone

        layers = dict([*self.backbone.named_children()])
        self.instance_layer = layers.get(instance_layer)
        self.pixel_layer = layers.get(pixel_layer)
        self._hook_registered = False

        self.MLP = nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048),
                                 nn.ReLU(), nn.Linear(2048, instance_shape))

        self.ConvMLP = nn.Sequential(nn.Conv2d(2048, 2048, 1, bias=False),
                                     nn.BatchNorm2d(2048), nn.ReLU(),
                                     nn.Conv2d(2048, pixel_shape, 1))

    def _hook(self, attr_name, _, __, output):
        setattr(self, attr_name, output)

    def register_hook(self):
        self.instance_layer.register_forward_hook(
            partial(self._hook, 'instance_feature'))
        self.pixel_layer.register_forward_hook(
            partial(self._hook, 'pixel_feature'))

    def forward(self, x):
        if self._hook_registered is False:
            self._hook_registered = True
            self.register_hook()

        self.backbone(x)
        self.instance_feature = self.instance_feature.squeeze()
        ins = self.MLP(self.instance_feature)
        pix = self.ConvMLP(self.pixel_feature)

        return ins, pix


class PPM(nn.Module):
    def __init__(self, chan=256, layers=1, gamma=2):
        super().__init__()

        self.gamma = gamma

        if layers == 0:
            self.transform = nn.Identity()
        elif layers == 1:
            self.transform = nn.Conv2d(chan, chan, 1)
        elif layers == 2:
            self.transform = nn.Sequential(nn.Conv2d(chan, chan, 1),
                                           nn.BatchNorm2d(chan), nn.ReLU(),
                                           nn.Conv2d(chan, chan, 1))
        else:
            raise ValueError('layers must be one of 0, 1, or 2')

    def forward(self, x):
        xi = x[:, :, :, :, None, None]  # (B, C, H, W, 1, 1)
        xj = x[:, :, None, None, :, :]  # (B, C, 1, 1, H, W)
        s = F.relu(F.cosine_similarity(xi, xj, dim=1))**self.gamma  # (B, H, W, H, W)

        g = self.transform(x)
        out = torch.einsum('bijhw, bchw -> bcij', s, g)
        return out


@torch.no_grad()
class Momentum():
    def __init__(self, init_momentum, max_step):
        self.init_momentum = init_momentum
        self.beta = init_momentum
        self.max_step = max_step
        self.crt_step = 0

    def update(self, encoder, mom_encoder):
        for params, mom_params in zip(encoder.parameters(),
                                      mom_encoder.parameters()):
            old, new = mom_params.data, params.data
            mom_params.data = old * self.beta + (1 - self.beta) * new
        self._update_mom()

    def _update_mom(self):
        self.crt_step = self.crt_step + 1
        self.beta = 1 - (1 - self.init_momentum) * (
            np.cos(np.pi * self.crt_step / self.max_step) + 1) / 2

# class InstanceLoss(nn.Module):
#     def __init__(self, temp=0.07):
#         super(InstanceLoss, self).__init__()
#         self.temp = temp
#         self.base = nn.CrossEntropyLoss()

#     def forward(self, x1, x2):
#         x1 = x1[:, None, ...]  # (B, 1, N)
#         x2 = x2[None, :, ...]  # (1, B, N)
#         s = F.cosine_similarity(x1, x2, dim=2) / self.temp  # (B, B)
#         target = torch.arange(s.shape[0]).cuda(s.device)
#         return self.base(s, target)


class PixConsistLoss(nn.Module):
    def __init__(self):
        super(PixConsistLoss, self).__init__()

    def forward(self, x1, x2, mask):
        x1 = torch.flatten(x1, -2)[..., :, None]  # (B, C, H*W, 1)
        x2 = torch.flatten(x2, -2)[..., None, :]  # (B, C, 1, H*W)
        s = F.cosine_similarity(x1, x2, dim=1)  # (B, H*W, H*W)
        result = torch.where(mask, s, torch.zeros(s.shape).cuda(s.device))  # (B, H*W, H*W)

        result = result.sum(dim=(-1, -2))
        matches_per_image = mask.sum(dim=(-1, -2))

        result = result.masked_select(matches_per_image > 0)
        matches_per_image = matches_per_image.masked_select(matches_per_image > 0)
        result = result / matches_per_image

        return 1 - result.mean()


class PixPro(nn.Module):
    def __init__(
        self,
        backbone,
        max_step=200,
        # alpha=1,
        thres=0.7,
        # temp=0.07,
        init_momentum=0.99,
        ppm_layers=1,
        ppm_gamma=2,
    ):
        super(PixPro, self).__init__()
        self.encoder = Wapper(backbone)
        self.mom_encoder = copy.deepcopy(self.encoder)

        self.ppm = PPM(chan=256, layers=ppm_layers, gamma=ppm_gamma)
        self.momentum = Momentum(init_momentum=init_momentum,
                                 max_step=max_step)
        # self.alpha = alpha
        self.thres = thres
        # self.inscriteria = InstanceLoss(temp=temp)
        self.pixcriteria = PixConsistLoss()

    def set_maxstep(self, step):
        self.momentum.max_step = step

    def update_momentum(self):
        with torch.no_grad():
            self.momentum.update(self.encoder, self.mom_encoder)

    def forward(self, views):
        size = views['size']
        view1 = views['view1']
        view2 = views['view2']
        grid1 = views['grid1']
        grid2 = views['grid2']

        instance1, x1 = self.encoder(view1)
        instance2, x2 = self.encoder(view2)
        y1 = self.ppm(x1)
        y2 = self.ppm(x2)

        with torch.no_grad():
            instance1_prime, x1_prime = self.mom_encoder(view1)
            instance2_prime, x2_prime = self.mom_encoder(view2)

        grid1 = F.interpolate(grid1, (7, 7), mode='bilinear')
        grid2 = F.interpolate(grid2, (7, 7), mode='bilinear')

        # instance_loss = (self.inscriteria(instance1, instance2_prime) +
        #                  self.inscriteria(instance2, instance1_prime)) / 2

        grid1 = torch.flatten(grid1, -2)
        grid2 = torch.flatten(grid2, -2)
        grid1 = grid1[..., :, None]  # (B, 2, H*W, 1)
        grid2 = grid2[..., None, :]  # (B, 2, 1, H*W)
        dist = torch.norm(grid1 - grid2, dim=1)  # (B, H*W, H*W)
        size = size.view(-1, 1, 1)  # (B, 1, 1)
        dist = dist / size * 7
        mask = (dist < self.thres)
        mask_t = mask.permute(0, 2, 1)

        pixel_loss = (self.pixcriteria(y1, x2_prime, mask) +
                      self.pixcriteria(y2, x1_prime, mask_t))

        if torch.isnan(pixel_loss):
            return -1

        # return pixel_loss + self.alpha * instance_loss
        return pixel_loss
