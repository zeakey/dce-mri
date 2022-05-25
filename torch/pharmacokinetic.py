import torch, math, os
import numpy as np
from einops import rearrange
from vlkit.lrscheduler import CosineScheduler, MultiStepScheduler
from torch.utils.tensorboard import SummaryWriter


def parker_aif(a1, a2, t1, t2, sigma1, sigma2, alpha, beta, s, tau, t):
    """
    The parker artierial input function
    """
    aif = a1 / (sigma1 * math.sqrt(2 * math.pi)) * (-(t - t1) ** 2 / (2 * sigma1 ** 2)).exp() + \
          a2 / (sigma2 * math.sqrt(2 * math.pi)) * (-(t - t2) ** 2 / (2 * sigma2 ** 2)).exp() + \
          alpha * (-beta * t).exp() / (1 + (-s * (t - tau)).exp())
    aif[t < 0] = 0
    return aif


def tofts(ktrans, kep, t0, t, aif_t, aif_cp):
    """
    Tofts model

    ktrans, kep, t0: [h, w]
    t: [h, w, t1]
    aif_t, aif_cp: [h, w, t2]
    For batch process, here I convert the individual convolution where each case has its own
    kernel to a grouped 1-D convolution.
    """
    h, w = ktrans.shape
    t1, t2 = t.shape[2], aif_cp.shape[2]
    dt = aif_t[:, :, 1] - aif_t[:, :, 0]

    assert t.shape[:2] == torch.Size([h, w])
    assert aif_t.shape[:2] == torch.Size([h, w])
    assert aif_cp.shape[:2] == torch.Size([h, w])


    # impulse response
    impulse = ktrans.unsqueeze(dim=-1) * (-kep.unsqueeze(dim=-1) * aif_t).exp()

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
    gridx = ((t - t0.view(h, w, 1)) / dt.view(h, w, 1))
    gridx = rearrange(gridx, 'h w t1 -> 1 (h w) t1 1')
    gridy = torch.arange(h*w).view(1, h*w, 1, 1).repeat(1, 1, t1, 1).to(gridx)
    # normalize to [-1, 1]
    gridx = gridx / (t2 - 1) * 2 -1
    if h * w == 1:
        gridy.fill_(-1)      
    elif h * w > 1:
        gridy = gridy / (h * w - 1) * 2 -1
    grid = torch.cat((gridx, gridy), dim=-1) # shape: [1, h*w, t1, 2]
    interp = torch.nn.functional.grid_sample(conv, grid, align_corners=True) # shape: [1, 1, h*w, t1]
    # matlab mean(interp) = 0.3483
    interp = rearrange(interp, '1 1 (h w) t1 -> h w t1', h=h, w=w, t1=t1)
    return interp


def fit_tofts(
    t, ct, aif_t, aif_cp,
    max_iter=250, max_lr=2,
    init_params=[0.3, 1, 0],
    tensorboard=None
    ):

    h, w, frames = ct.shape

    params = torch.ones(h, w, 3).to(ct) * torch.tensor(init_params).view(1, 1, 3).to(t)
    params.requires_grad = True

    def bound_loss(params, lower_bound, upper_bound):
        assert lower_bound.numel() == 3 and upper_bound.numel() == 3 and params.shape[2] == 3
        lower_bound = lower_bound.view(1, 1, 3)
        upper_bound = upper_bound.view(1, 1, 3)

        lower_loss = (lower_bound - params) * (params < lower_bound)
        upper_loss = (params - upper_bound) * (params > upper_bound)
        loss = (lower_loss + upper_loss).mean()
        return loss

    lower_bound = torch.tensor([1e-3, 1e-3, 0]).to(t)
    upper_bound = torch.tensor([5, 1, 0.2544]).to(t)

    # lrscheduler = MultiStepScheduler(epoch_iters=100, milestones=[2, 3, 4], gamma=0.1, base_lr=max_lr, warmup_iters=20)
    lrscheduler = CosineScheduler(epoch_iters=max_iter, epochs=1, max_lr=max_lr, min_lr=max_lr*1e-3, warmup_iters=20)
    optimizer = torch.optim.SGD(params=[params], lr=0, momentum=0.9)
    # optimizer = torch.optim.Adam(params=[params], lr=max_lr)

    if tensorboard is not None:
        os.makedirs(tensorboard, exist_ok=True)
        tensorboard = SummaryWriter(tensorboard)

    for it in range(max_iter):
        optimizer.zero_grad()
        ct_hat = tofts(params[:, :, 0], params[:, :, 1], params[:, :, 2], t, aif_t, aif_cp)

        loss = torch.nn.functional.mse_loss(ct_hat, ct, reduction='none').sum(dim=2)
        loss.sum().backward()
        lr = lrscheduler.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        # projected gradient descent, apply constrains
        optimizer.param_groups[0]['params'][0].data[:, :, 0].clamp_(1e-3, 5)
        optimizer.param_groups[0]['params'][0].data[:, :, 1].clamp_(1e-3, 1)
        optimizer.param_groups[0]['params'][0].data[:, :, 2].clamp_(0, 0.25)

        # logger
        if tensorboard is not None:
            tensorboard.add_scalar('lr', lr, it)
            tensorboard.add_scalar('loss', loss.mean().item(), it)
        print('iter {it:03d}, loss={loss:.2e}, lr={lr:.2e}'.format(it=it, loss=loss.mean().item(), lr=lr))
    return params.detach(), loss.detach()


def get_ct_curve(params, t, aif_t, aif_cp):
    ktrans = params[0].view(1, 1)
    kep = params[1].view(1, 1)
    t0 = params[2].view(1, 1)
    return tofts(ktrans, kep, t0, t, aif_t, aif_cp).view(-1)


if __name__ == '__main__':
    from scipy.io import loadmat, savemat
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from vlkit.image import norm01
    import time

    device = torch.device('cuda')

    def np2torch(data):
        for k, v in data.items():
            if isinstance(v, np.ndarray): data[k] = torch.tensor(v).float().to(device=device)
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
    ktrans = data['a'].squeeze()[0].view(1, 1)
    kep = data['a'].squeeze()[1].view(1, 1)
    t0 = data['a'].squeeze()[2].view(1, 1)
    t = data['t'].squeeze().view(1, 1, -1)
    aif_t = data['aif'][:, 0].view(1, 1, -1)
    aif_cp = data['aif'][:, 1].view(1, 1, -1)
    interp = tofts(ktrans.repeat(2, 2), kep.repeat(2, 2), t0.repeat(2, 2), t.repeat(2, 2, 1), aif_t.repeat(2, 2, 1), aif_cp.repeat(2, 2, 1))
    interp1 = tofts(ktrans, kep, t0, t, aif_t, aif_cp)
    # assert torch.all(interp[0, 0, :] == interp1)

    plt.plot(interp[0, 0, :].squeeze().cpu())
    plt.plot(data['F'].squeeze().cpu())
    plt.legend(['Torch', 'Matlab'])
    plt.savefig('tofs.pdf')
    plt.close() 

    # fit tofts
    # fit a sisngle voxel
    data = loadmat('../tmp/fit_cc.mat')
    data = np2torch(data)
    ct = data['cc'].squeeze().view(1,1, -1)
    t = data['time'].view(1, 1, -1)
    params, _, = fit_tofts(
        t, ct, aif_t, aif_cp,
        max_lr=1e-3,
        tensorboard='tensorboard/single-voxel-cosine-lr1e-3'
    )

    matlab_reconsructed_curve = data['ctTmp_hat'].squeeze()
    torch_reconsructed_curve = tofts(params[:, :, 0], params[:, :, 1], params[:, :, 2], t, aif_t, aif_cp).squeeze()

    loss_matlab = torch.nn.functional.mse_loss(matlab_reconsructed_curve, data['cc'].squeeze(), reduction='sum')
    loss_torch = torch.nn.functional.mse_loss(torch_reconsructed_curve, data['cc'].squeeze(), reduction='sum')

    plt.plot(data['cc'].squeeze().cpu())
    plt.plot(matlab_reconsructed_curve.cpu())
    plt.plot(torch_reconsructed_curve.cpu())
    plt.legend(['Obserevation', 'Matlab Reconstruction', 'Torch Reconstruction'])
    plt.tight_layout()
    plt.savefig('fit_a_voxel.pdf')
    plt.close() 

    # fit an entire volumn
    data = loadmat('../tmp/fit_tofts.mat')
    data = np2torch(data)
    ct = data['ct'].squeeze()
    # ct1 = ct[:, :, 0:10, :]
    ct1 = rearrange(ct, 'h w s f -> h (w s) f')
    h, w, _ = ct1.shape
    t = data['time'].view(1, 1, -1)
    tic = time.time()
    params, loss = fit_tofts(
        t.repeat(h, w, 1),
        ct1,
        aif_t.repeat(h, w, 1),
        aif_cp.repeat(h, w, 1),
        max_lr=1e-4,
        max_iter=100,
        init_params=[0.1172, 0.4666, 0.0044],
        tensorboard='tensorboard/cosine-lr1e-4'
    )
    print("Done, ETA=%.3f" % (time.time() - tic))
    params = rearrange(params, 'h (w s) p -> h w s p', h=160, w=160, s=20, p=3)

    # debug outliers
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    idx = params[:, :, :, 0].argmax().item()
    maxv = params[:, :, :, 1].flatten()[idx].item()
    params1 = torch.stack([
        params[:, :, :, 0].flatten()[idx],
        params[:, :, :, 1].flatten()[idx],
        params[:, :, :, 2].flatten()[idx]
    ])
    ct1_hat = get_ct_curve(params1, t, aif_t, aif_cp).cpu().flatten()
    axes[0].plot(ct1.reshape(-1, 75)[idx].flatten().cpu().numpy())
    axes[0].plot(ct1_hat)
    axes[0].set_title('Max Ktrans=%.3e' % maxv)

    idx = params[:, :, :, 1].argmax().item()
    maxv = params[:, :, :, 1].flatten()[idx].item()
    params1 = torch.stack([
        params[:, :, :, 0].flatten()[idx],
        params[:, :, :, 1].flatten()[idx],
        params[:, :, :, 2].flatten()[idx]
    ])
    ct1_hat = get_ct_curve(params1, t, aif_t, aif_cp).cpu().flatten()
    axes[1].plot(ct1.reshape(-1, 75)[idx].flatten().cpu().numpy())
    axes[1].plot(ct1_hat)
    axes[1].set_title('Max Kep=%.3e' % maxv)
    plt.savefig('debug_max_params.pdf')
    plt.close()


    # compare results of Matlab and Torch
    data = loadmat('../tmp/volumn_results.mat')
    matlab_ktrans = data['B'][:, :, :, 0]
    torch_ktrans = params.cpu().numpy()[:, :, :, 0]

    n = 10
    fig, axes = plt.subplots(n, 2, figsize=(4, 20))
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_title("Matlab")
    axes[0, 1].set_title("Torch")
    for i in range(0, 20, 2):
        axes[i//2, 0].imshow(matlab_ktrans[:, :, i])
        axes[i//2, 1].imshow(torch_ktrans[:, :, i])
        axes[i//2, 0].set_ylabel("slice#%.2d"%i)
    plt.tight_layout()
    plt.savefig('ktrans.pdf')
    plt.close() 

    matlab_kep = data['B'][:, :, :, 1]
    torch_kep = params.cpu().numpy()[:, :, :, 1]
    # # remove outliers
    # torch_kep[torch_kep >= torch_kep.mean()*2] = 0
    n = 10
    fig, axes = plt.subplots(n, 2, figsize=(4, 20))
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_title("Matlab")
    axes[0, 1].set_title("Torch")
    for i in range(0, 20, 2):
        axes[i//2, 0].imshow(matlab_kep[:, :, i])
        axes[i//2, 1].imshow(torch_kep[:, :, i])
        axes[i//2, 0].set_ylabel("slice#%.2d"%i)
    plt.tight_layout()
    plt.savefig('kep.pdf')
    plt.close() 