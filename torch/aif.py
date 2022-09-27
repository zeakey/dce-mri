import math, torch, einops
from vlkit.ops import conv1d


def parker_aif(a1, a2, t1, t2, sigma1, sigma2, alpha, beta, s, tau, t):
    """
    The parker artierial input function
    """
    cb = a1 / (sigma1 * math.sqrt(2 * math.pi)) * (-(t - t1) ** 2 / (2 * sigma1 ** 2)).exp() + \
          a2 / (sigma2 * math.sqrt(2 * math.pi)) * (-(t - t2) ** 2 / (2 * sigma2 ** 2)).exp() + \
          alpha * (-beta * t).exp() / (1 + (-s * (t - tau)).exp())
    cb[t < 0] = 0
    return cb


def biexp_aif(a1, a2, m1, m2, t):
    cb = a1 * (-m1 * t).exp() + a2 * (-m2 * t).exp()
    cb[t < 0] = 0
    return cb


def get_aif(aif: str, acquisition_time: torch.Tensor, max_base: int, hct: float=0.42, device=torch.device('cpu')):
    assert aif in ['parker', 'weinmann', 'fh']
    aif_t = torch.arange(0, acquisition_time[-1], 1/60).to(acquisition_time)

    if aif == 'parker':
        aif_cp = parker_aif(
            a1=0.809,
            a2=0.330,
            t1=0.17046,
            t2=0.365,
            sigma1=0.0563,
            sigma2=0.132,
            alpha=1.050,
            beta=0.1685,
            s=38.078,
            tau=0.483,
            t=aif_t - (acquisition_time[max_base] / (1 / 60)).ceil() * (1 / 60)
        ) / (1 - hct)
    elif aif == 'weinmann':
        aif_cp = biexp_aif(
            3.99,
            4.78,
            0.144,
            0.011,
            aif_t - (acquisition_time[max_base] / (1 / 60)).ceil() * (1 / 60)
        ) / (1 - hct)
    elif aif == 'fh':
        aif_cp = biexp_aif(
            24,
            6.2,
            3.0,
            0.016,
            aif_t
        ) / (1 - hct)
    return aif_cp.to(device), aif_t.to(device)


def dispersed_aif(aif, aif_t, beta):
    """
    dispersed AIF function
    aif: AIF function, [NxD] tensor
    aif_t: AIF time, D dimensional vector
    beta: beta, N dimentional vector
    """
    assert aif.ndim == 1 or aif.ndim == 2
    assert aif.shape == aif_t.shape

    orig_shape = list(beta.shape)
    n = beta.numel()
    beta = beta.view(-1, 1)

    if aif.ndim == 1:
        aif = einops.repeat(aif, 'd -> n d', n=n)
        aif_t = einops.repeat(aif_t, 'd -> n d', n=n)
    else:
        assert aif.shape[0] == n and aif_t.shape[0] == n

    dt = aif_t[0, 1] - aif_t[0, 0]
    d = aif.shape[1]

    ht =  aif_t.neg().div(beta).exp().div(beta)
    ht = ht / ht.sum(dim=1, keepdim=True) / dt

    dispersed_aif =  conv1d(aif, ht) * dt
    orig_shape.append(dispersed_aif.shape[-1])
    dispersed_aif = dispersed_aif.view(orig_shape)
    return dispersed_aif[..., :d]


if __name__ == '__main__':
    import matplotlib, sys
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    acquisition_time = torch.linspace(0, 5.28, 75)

    pk_aif, aif_t = get_aif('parker', acquisition_time=acquisition_time, max_base=6)
    beta = torch.linspace(1e-2, 1, 10)
    d_aif = dispersed_aif(pk_aif, aif_t, beta=beta)

    legend = []
    plt.plot(pk_aif)
    legend.append('Parker AIF')

    for i in range(10):
        plt.plot(d_aif[i,])
        legend.append('beta=%.2e' % beta[i])
    plt.legend(legend)
    plt.grid()
    plt.savefig('dispersed_aif.pdf')

    dispersed_data = loadmat('../tmp/dispersed_aif.mat')
    d_aif_matlab = torch.tensor(dispersed_data['aif_dispersed'][:, 1, :]).t()
    print(d_aif_matlab.shape)

    beta = torch.tensor(dispersed_data['beta_all'], dtype=torch.float).view(-1)
    aif = torch.tensor(dispersed_data['aif1'], dtype=torch.float)
    aif_t = aif[:, 0]
    aif = aif[:, 1]

    d_aif = dispersed_aif(aif, aif_t, beta=beta)
    print(aif.shape)
    print(aif.sum())
    print(d_aif.shape)
    print(d_aif.sum(dim=1))
    print(d_aif_matlab.sum(dim=1))

    sys.exit()

    aif_time = torch.arange(0, 7, 1/60) - 0.2
    hct = 0.42
    # cp_weinmann = biexp_aif(3.99, 4.78, 0.144, 0.011, tp-ceil(0.25/tdel)*tdel)/(1-Hct); % Weinmann
    # cp_fh = biexp_aif(24, 6.2, 3.0, 0.016, tp-ceil(0.25/tdel)*tdel)/(1-Hct); % Fritz-Hans
    weinmann_cp = biexp_aif(3.99, 4.78, 0.144, 0.011, aif_time) / (1 - hct)
    fh_cp = biexp_aif(24, 6.2, 3.0, 0.016, aif_time) / (1 - hct)

    parker_cp = parker_aif(
        a1=0.809, a2=0.330,
        t1=0.17046, t2=0.365,
        sigma1=0.0563, sigma2=0.132,
        alpha=1.050, beta=0.1685, s=38.078, tau=0.483,
        t=aif_time,
        ) / (1 - hct)
    
    plt.plot(parker_cp)
    plt.plot(weinmann_cp)
    plt.plot(fh_cp)
    plt.legend(['Parker AIF', 'Weinmann AIF', 'Fritz-Hans AIF'])
    plt.savefig('aif.pdf')
    plt.savefig('aif.jpg')

