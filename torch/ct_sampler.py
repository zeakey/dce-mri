import torch
import torch.nn as nn
from torch.distributions import Beta, Uniform


class CTSampler(object):
    def __init__(self):
        super().__init__()
        self.ktrans_sampler = Beta(concentration1=1.3, concentration0=13)
        self.kep_sampler = Beta(concentration1=1.3, concentration0=20)
        self.t0_sampler = Uniform(low=0, high=0.5)

    def sample(self, n):
        ktrans = self.ktrans_sampler.sample((n, 1))
        kep = self.kep_sampler.sample((n, 1))
        t0 = self.t0_sampler.sample((n, 1))
        return torch.cat((ktrans, kep, t0), dim=1)