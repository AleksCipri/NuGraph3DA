#following SIDDA paper
# https://www.kernel-operations.io/geomloss/

import torch
from functools import partial
from torch import Tensor
import geomloss


class Sinkhorn(torch.nn.Module):
    def __init__(self, blur):
        super().__init__()
        self.blur = blur  # Store blur in self

    def forward(self, x, y, blur=None):
        if blur is None:
            blur = self.blur  # Use the default blur if not provided

        # Create a new SamplesLoss instance with the correct blur each time
        loss = geomloss.SamplesLoss("sinkhorn", blur=blur, scaling=0.9, reach=None)
        return loss(x, y)  # ✅ Now works correctly


@staticmethod
def kl_divergence(p, q):
    epsilon = 1e-6
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return torch.sum(p * torch.log(p / q), dim=-1)

@staticmethod
def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd

@staticmethod
def jensen_shannon_distance(p, q):
    jsd = jensen_shannon_divergence(p, q)
    jsd = torch.clamp(jsd, min=0.0)
    return torch.sqrt(jsd)


