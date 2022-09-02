from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Siamvgg']


class Siamvgg(nn.Module):

    def __init__(self, out_scale=0.001):
        super(Siamvgg, self).__init__()
        self.out_scale = out_scale
        self.bn_adjust = nn.BatchNorm2d(1)

    def forward(self, z, x):
        return self._fast_xcorr(z, x)
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation

        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        out=self.bn_adjust(out)

        return out
