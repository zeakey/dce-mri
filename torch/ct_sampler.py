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
    def __init__(self, config, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.ktrans_scale = config.ktrans_sampler.pop('scale')
        self.kep_scale = config.kep_sampler.pop('scale')
        self.t0_scale = config.t0_sampler.pop('scale')

        self.ktrans_sampler = build_sampler(config.ktrans_sampler)
        self.kep_sampler = build_sampler(config.kep_sampler)
        self.t0_sampler = build_sampler(config.t0_sampler)

        if hasattr(config, 'beta_sampler'):
            self.beta_sampler = build_sampler(config.beta_sampler)
        else:
            self.beta_sampler = False

    def sample(self, n):
        results = dict()
        results["ktrans"] = self.ktrans_sampler.sample((n,)).to(self.device) * self.ktrans_scale
        results["kep"] = self.kep_sampler.sample((n,)).to(self.device) * self.kep_scale
        results["t0"] = self.t0_sampler.sample((n,)).to(self.device) * self.t0_scale
        if self.beta_sampler:
            results["beta"] = self.beta_sampler.sample((n,)).to(self.device)
        return results



