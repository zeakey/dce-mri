import math, torch


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


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

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

