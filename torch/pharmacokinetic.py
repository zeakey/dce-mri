import torch, math, sys
import numpy as np
from einops import rearrange


def parker_aif(a1, a2, t1, t2, sigma1, sigma2, alpha, beta, s, tau, time):
    """
    The parker artierial input function
    """
    aif = a1 / (sigma1 * math.sqrt(2 * math.pi)) * (-(time - t1) ** 2 / (2 * sigma1 ** 2)).exp() + \
          a2 / (sigma2 * math.sqrt(2 * math.pi)) * (-(time - t2) ** 2 / (2 * sigma2 ** 2)).exp() + \
          alpha * (-beta * time).exp() / (1 + (-s * (time - tau)).exp())
    aif[time < 0] = 0
    return aif


def tofts(ktrans, kep, t0, time, aif_time, aif_cp):
    """
    Tofts model

    ktrans, kep, t0: [h, w]
    time: [h, w, t1]
    aif_time, aif_cp: [h, w, t2]
    For batch process, here I convert the individual convolution where each case has its own
    kernel to a grouped 1-D convolution.
    """
    h, w = ktrans.shape
    t1, t2 = time.shape[2], aif_cp.shape[2]
    dt = aif_time[:, :, 1] - aif_time[:, :, 0]

    # impulse response
    impulse = ktrans.unsqueeze(dim=-1) * (-kep.unsqueeze(dim=-1) * aif_time).exp()

    # rearrange shapes for 1-D convolution
    # see https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html
    # for document to `torch.nn.functional.conv1d`
    aif_cp = rearrange(aif_cp, 'h w t2 -> 1 (h w) t2')
    impulse = rearrange(impulse, 'h w t2 -> (h w) 1 t2')
    # convolve the impulse response with AIF
    conv = torch.nn.functional.conv1d(input=aif_cp, weight=impulse.flip(dims=(-1,)), padding=t2-1, groups=h*w)[:, :, :t2]
    conv = rearrange(conv, '1 (h w) t2 -> h w t2', h=h, w=w) * dt.view(h, w, 1)
    
    # interpolate
    conv = rearrange(conv, 'h w t2 -> 1 1 (h w) t2', h=h, w=w, t2=t2)
    gridx = ((time - t0.view(h, w, 1)) / dt.view(h, w, 1))
    gridx = rearrange(gridx, 'h w t1 -> 1 (h w) t1 1')
    gridy = torch.arange(h*w).view(1, 4, 1, 1).repeat(1, 1, t1, 1).to(gridx)
    # normalize to [-1, 1]
    gridx = gridx / (t2 - 1) * 2 -1
    gridy = gridy / (h * w - 1) * 2 -1
    grid = torch.cat((gridx, gridy), dim=-1) # shape: [n, h, w, 2]
    interp = torch.nn.functional.grid_sample(conv, grid, align_corners=True) # shape: [n, c, h, w]
    # matlab mean(interp) = 0.3483
    return interp


if __name__ == '__main__':
    from scipy.io import loadmat
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    def np2torch(data):
        for k, v in data.items():
            if isinstance(v, np.ndarray): data[k] = torch.tensor(v)
        return data

    data = loadmat('../tmp/parker_aif.mat')
    data = np2torch(data)
            
    aif = parker_aif(
        data['A1'], data['A2'], data['T1'], data['T2'],
        data['sigma1'], data['sigma2'], data['alpha'],
        data['beta'], data['s'], data['tau'], data['time']
    )
    print('aif.std() =', aif.std().item())

    data = loadmat('../tmp/fun_Tofts.mat')
    data = np2torch(data)
    ktrans = data['a'].squeeze()[0].view(1, 1).repeat(2, 2)
    kep = data['a'].squeeze()[1].view(1, 1).repeat(2, 2)
    t0 = data['a'].squeeze()[2].view(1, 1).repeat(2, 2)
    time = data['t'].squeeze().view(1, 1, -1).repeat(2, 2, 1)
    aif_time = data['aif'][:, 0].view(1, 1, -1).repeat(2, 2, 1)
    aif_cp = data['aif'][:, 1].view(1, 1, -1).repeat(2, 2, 1)
    interp = tofts(ktrans, kep, t0, time, aif_time, aif_cp)

    plt.plot(interp[0,0,1, :].squeeze())
    plt.plot(data['F'].squeeze())
    plt.legend(['Torch', 'Matlab'])
    plt.savefig('tofs.pdf')