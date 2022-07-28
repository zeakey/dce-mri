import torch
import torch.nn as nn
from torch.distributions import Beta, Uniform
from pharmacokinetic import tofts


def build_sampler(cfg):
    distr = eval(cfg.type)
    kwrags = dict()
    for k, v in cfg.items():
        if k != 'type':
            kwrags[k] = v
    distr = distr(**kwrags)
    return distr


class CTSampler(object):
    def __init__(self, config):
        super().__init__()
        self.ktrans_scale = config.ktrans_sampler.pop('scale')
        self.kep_scale = config.kep_sampler.pop('scale')
        self.t0_scale = config.t0_sampler.pop('scale')

        self.ktrans_sampler = build_sampler(config.ktrans_sampler)
        self.kep_sampler = build_sampler(config.kep_sampler)
        self.t0_sampler = build_sampler(config.t0_sampler)

    def sample(self, n):
        ktrans = self.ktrans_sampler.sample((n, 1)) * self.ktrans_scale
        kep = self.kep_sampler.sample((n, 1)) * self.kep_scale
        t0 = self.t0_sampler.sample((n, 1)) * self.t0_scale
        return torch.cat((ktrans, kep, t0), dim=1)


def generate_data(ktrans, kep, t0, aif_t, aif_cp, t):
    batch_size = ktrans.shape[0]
    t = t.view(1, 1, -1).repeat(batch_size, 1, 1)
    ct = tofts(ktrans, kep, t0, t, aif_t, aif_cp)
    noice = torch.randn(ct.shape, device=ct.device) / 4
    ct += ct * noice
    ct[ct < 0] = 0
    return ct
