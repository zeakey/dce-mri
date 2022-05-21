import torch, math
import numpy as np


def parker_aif(a1, a2, t1, t2, sigma1, sigma2, alpha, beta, s, tau, time):
    """
    The parker artierial input function
    """
    aif = a1 / (sigma1 * math.sqrt(2 * math.pi)) * (-(time - t1) ** 2 / (2 * sigma1 ** 2)).exp() + \
          a2 / (sigma2 * math.sqrt(2 * math.pi)) * (-(time - t2) ** 2 / (2 * sigma2 ** 2)).exp() + \
          alpha * (-beta * time).exp() / (1 + (-s * (time - tau)).exp())
    aif[time < 0] = 0
    return aif


def tofts(ktrans, kep, t0, time, aif):
    """
    Tofts model
    """
    return None


if __name__ == '__main__':
    from scipy.io import loadmat
    data = loadmat('../tmp/parker_aif.mat')
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            # print(k, v)
            data[k] = torch.tensor(v)
            
    aif = parker_aif(
        data['A1'], data['A2'], data['T1'], data['T2'],
        data['sigma1'], data['sigma2'], data['alpha'],
        data['beta'], data['s'], data['tau'], data['time']
    )
    print(aif.std())