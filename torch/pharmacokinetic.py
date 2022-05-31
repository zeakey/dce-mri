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
    assert ktrans.ndim == 2 and kep.ndim == 2 and t0.ndim == 2
    h, w = ktrans.shape
    t1, t2 = t.shape[2], aif_t.shape[2]
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


def fit_slice(
    t, ct, aif_t, aif_cp,
    max_iter=250, max_lr=2,
    init_params=None,
    tensorboard=None
    ):

    h, w, frames = ct.shape

    if init_params is not None:
        params = torch.ones(h, w, 3).to(ct) * torch.tensor(init_params).view(1, 1, 3).to(t)
    else:
        params = torch.ones(h, w, 3).to(ct)
        params[:, :, 0] = torch.nn.init.normal(params[:, :, 0], mean=2.5, std=1).clamp(0, 5)
        params[:, :, 1] = torch.nn.init.normal(params[:, :, 1], mean=2, std=5).clamp(0, 50)
        params[:, :, 2] = torch.nn.init.normal(params[:, :, 2], mean=0.125, std=0.1).clamp(0, 0.25)
    params.requires_grad = True

    lrscheduler = CosineScheduler(epoch_iters=max_iter, epochs=1, max_lr=max_lr, min_lr=max_lr*1e-5, warmup_iters=20)
    optimizer = torch.optim.RMSprop(params=[params], lr=0, momentum=0.9)


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
        optimizer.param_groups[0]['params'][0].data[:, :, 0].clamp_(0, 5)
        optimizer.param_groups[0]['params'][0].data[:, :, 1].clamp_(0, 50)
        optimizer.param_groups[0]['params'][0].data[:, :, 2].clamp_(0, 0.25)

        # logger
        if tensorboard is not None:
            tensorboard.add_scalar('lr', lr, it)
            tensorboard.add_scalar('loss', loss.mean().item(), it)
        if it % (max_iter // 50) == 0 or it == (max_iter - 1):
            print('iter {it:03d}, loss={loss:.2e}, lr={lr:.2e}. grad1={grad1:.2e} grad2={grad2:.2e} grad3={grad3:.2e}'.format(
                it=it, loss=loss.mean().item(), lr=lr,
                grad1 = optimizer.param_groups[0]['params'][0].grad[:, :, 0].abs().mean().item(),
                grad2 = optimizer.param_groups[0]['params'][0].grad[:, :, 1].abs().mean().item(),
                grad3 = optimizer.param_groups[0]['params'][0].grad[:, :, 2].abs().mean().item()))
    return params.detach(), loss.detach()



def process_patient(time, ct, aif_t, aif_cp, max_iter=500, max_lr=1e-3, work_dir=None):
    h, w, slices, frames = ct.shape
    center = ct[h//2-6:h//2+6, w//2-6 : w//2+6, slices//2-1, :].mean(dim=(0,1), keepdim=True)
    time = time.view(1, 1, -1)
    aif_t = aif_t.view(1, 1, -1)
    aif_cp = aif_cp.view(1, 1, -1)
    init_params, _ = fit_slice(
        time, center, aif_t, aif_cp,
        max_iter=max_iter, max_lr=max_lr,
        init_params=[0.3, 1, 0],
        tensorboard=None
    )
    init_params = init_params.squeeze().cpu().tolist()
    # ---------------------------------------------------
    # for debug
    x, y = torch.randint(min(h, w), (2,)).tolist()
    s = torch.randint(slices, (1,)).item()
    ct1 = ct[y:y+1, x:x+1, s, :]
    param1, loss1 = fit_slice(
        time, ct1, aif_t, aif_cp,
        max_iter=max_iter, max_lr=max_lr,
        init_params=init_params,
        tensorboard=None)
    # ---------------------------------------------------
    h1, w1 = h, w * slices
    params, loss = fit_slice(
        time.repeat(h1, w1, 1),
        rearrange(ct, 'h w s f -> h (w s) f'),
        aif_t.repeat(h1, w1, 1),
        aif_cp.repeat(h1, w1, 1),
        max_iter=max_iter, max_lr=max_lr,
        init_params=init_params,
        tensorboard=None if work_dir is None else osp.join(work_dir, 'tensorboard')
        )
    params = rearrange(params, 'h (w s) p -> h w s p', h=h, w=w, s=slices)
    loss = rearrange(loss, 'h (w s) -> h w s', h=h, w=w, s=slices)
    return params, loss



def get_ct_curve(params, t, aif_t, aif_cp):
    ktrans = params[0].view(1, 1)
    kep = params[1].view(1, 1)
    t0 = params[2].view(1, 1)
    return tofts(ktrans, kep, t0, t, aif_t, aif_cp).view(-1)


def compare_matlab_torch(matlab_param, torch_param, fig_filename):
    slices = matlab_param.shape[2]
    assert matlab_param.shape[2] == torch_param.shape[2]
    os.makedirs(osp.dirname(fig_filename), exist_ok=True)

    fig, axes = plt.subplots(slices, 2, figsize=(4, 20))
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_title("Matlab")
    axes[0, 1].set_title("Torch")
    for i in range(0, slices):
        axes[i, 0].imshow(matlab_param[:, :, i])
        axes[i, 1].imshow(torch_param[:, :, i])
        axes[i, 0].set_ylabel("slice#%.2d"%i)
    plt.tight_layout()
    plt.savefig(fig_filename)
    plt.close()


if __name__ == '__main__':
    from scipy.io import loadmat, savemat
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import pydicom
    import os.path as osp
    import time, sys
    from utils import write_dicom, write_dicom_meta

    def save_slices_to_dicom(data, dicom_dir, **kwargs):
        data = data.astype(np.float64)
        data = (data * 1000).astype(np.uint16)
        slices = data.shape[2]
        for i in range(slices):
            origin_dicom = '/data1/IDX_Current/dicom/10042_1_004D6Sy8/20160616/iCAD-MCC-Ktrans-FA-0-E_33009/IM-50068-%.4d.dcm' % (i+1)
            origin_dicom = pydicom.dcmread(origin_dicom)
            save_fn = osp.join(dicom_dir, 'slice-%.3d.dcm' % i)
            img = np.squeeze(data[:, :, i])
            thickness = 3.6
            write_dicom_meta(
                img,
                save_fn,
                ds=origin_dicom,
                **kwargs
            )


    # save_slices_to_dicom(np.random.randn(160, 160, 20), 'work_dirs/debug', SeriesDescription='PyTorch-Ktrans')
    # sys.exit()

    work_dir = 'work_dirs/RMSprop_lr1e-3~1e-8'

    device = torch.device('cuda')

    def np2torch(data):
        for k, v in data.items():
            if isinstance(v, np.ndarray): data[k] = torch.tensor(v).to(device=device).float()
        return data
    
    # Hct = 0.42;
    # cp_parker = parker_aif(0.809,0.330,0.17046,0.365,0.0563,0.132,1.050, ...
    # 0.1685,38.078,0.483,tp-ceil(time(maxBase)/tdel)*tdel)/(1-Hct);

    data = loadmat('../tmp/patient-0.mat')
    data = np2torch(data)

    time_dce = data['time_dce'] / 60
    aif_t = torch.arange(0, time_dce[-1].item(), 1/60, dtype=torch.float64).to(time_dce)

    hct = 0.42
    aif_cp = parker_aif(
        a1=0.809, a2=0.330,
        t1=0.17046, t2=0.365,
        sigma1=0.0563, sigma2=0.132,
        alpha=1.050, beta=0.1685, s=38.078, tau=0.483,
        t=aif_t - (time_dce[int(data['maxBase'].item())-1] * 60).ceil() / 60
        ) / (1 - hct)
    
    param, loss = process_patient(
        data['time_dce'] / 60,
        data['dce_ct'], aif_t,
        aif_cp,
        max_iter=500,
        work_dir=work_dir
    )
    matlab_param = torch.stack((data['ktrans'], data['kep'], data['t0']), dim=-1)
    compare_matlab_torch(data['ktrans'].cpu(), param[:,:,:,0].cpu(), osp.join(work_dir, 'ktrans.pdf'))
    compare_matlab_torch(data['kep'].cpu(), param[:,:,:,1].cpu(), osp.join(work_dir, 'kep.pdf'))
    compare_matlab_torch(data['t0'].cpu(), param[:,:,:,2].cpu(), osp.join(work_dir, 't0.pdf'))
    compare_matlab_torch(data['loss'].cpu(), loss.cpu(), osp.join(work_dir, 'loss.pdf'))

    # save results to dicom
    save_slices_to_dicom(data['ktrans'].cpu().numpy(), dicom_dir=osp.join(work_dir, 'dicom/matlab-ktrans/'), SeriesDescription='Matlab-Ktrans')
    save_slices_to_dicom(data['kep'].cpu().numpy(), dicom_dir=osp.join(work_dir, 'dicom/matlab-kep/'), SeriesDescription='Matlab-Kep')
    save_slices_to_dicom(data['t0'].cpu().numpy(), dicom_dir=osp.join(work_dir, 'dicom/matlab-t0/'), SeriesDescription='Matlab-T0')
    save_slices_to_dicom(data['loss'].cpu().numpy(), dicom_dir=osp.join(work_dir, 'dicom/matlab-loss/'), SeriesDescription='Loss-T0')
    #
    save_slices_to_dicom(param[:, :, :, 0].cpu().numpy(), dicom_dir=osp.join(work_dir, 'dicom/torch-ktrans/'), SeriesDescription='Torch-Ktrans')
    save_slices_to_dicom(param[:, :, :, 1].cpu().numpy(), dicom_dir=osp.join(work_dir, 'dicom/torch-kep/'), SeriesDescription='Torch-Kep')
    save_slices_to_dicom(param[:, :, :, 2].cpu().numpy(), dicom_dir=osp.join(work_dir, 'dicom/torch-t0/'), SeriesDescription='Torch-T0')
    save_slices_to_dicom(loss.cpu().numpy(), dicom_dir=osp.join(work_dir, 'dicom/torch-loss/'), SeriesDescription='Loss-T0')

    sys.exit()
    # fit tofts
    # fit a sisngle voxel
    data = loadmat('../tmp/fit_cc.mat')
    data = np2torch(data)
    ct = data['cc'].squeeze().view(1,1, -1)
    t = data['time'].view(1, 1, -1)
    params, loss_cc, = fit_slice(
        t, ct,
        aif_t.view(1, 1, -1),
        aif_cp.view(1, 1, -1),
        max_lr=1e-3,
        max_iter=1000,
        tensorboard=None
    )

    matlab_reconsructed_curve = data['cc_hat'].squeeze()
    torch_reconsructed_curve = tofts(params[:, :, 0], params[:, :, 1], params[:, :, 2], t, aif_t, aif_cp).squeeze()

    loss_matlab = torch.nn.functional.mse_loss(matlab_reconsructed_curve, data['cc'].squeeze(), reduction='sum')
    loss_torch = torch.nn.functional.mse_loss(torch_reconsructed_curve, data['cc'].squeeze(), reduction='sum')

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    matlab_params = data['p_init'].squeeze().cpu()
    axes[0].plot(data['cc'].squeeze().cpu())
    axes[0].plot(matlab_reconsructed_curve.cpu())
    axes[0].legend(['C(t)', 'Matlab (ktrans=%.3f, kep=%.3f, t0=%.3f)' % (matlab_params[0], matlab_params[1], matlab_params[2])])
    axes[0].set_title('Matlab')
    #
    axes[1].plot(data['cc'].squeeze().cpu())
    axes[1].plot(torch_reconsructed_curve.cpu())
    axes[1].legend(['C(t)', 'Torch (ktrans=%.3f, kep=%.3f, t0=%.3f)' % (params[:, :, 0], params[:, :, 1], params[:, :, 2])])
    axes[1].set_title('Torch')
    plt.tight_layout()
    plt.savefig(osp.join(work_dir, 'fit_a_voxel.pdf'))
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
    params, loss = fit_slice(
        t.repeat(h, w, 1),
        ct1,
        aif_t.repeat(h, w, 1),
        aif_cp.repeat(h, w, 1),
        max_lr=1e-3,
        max_iter=1000,
        init_params=None,
        tensorboard=osp.join(work_dir, 'tensorboard')
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
    plt.savefig(osp.join(work_dir, 'debug_max_params.pdf'))
    plt.close()


    # compare results of Matlab and Torch
    data = loadmat('../tmp/volumn_results.mat')

    

    params = params.cpu().numpy()
    np.save(osp.join(work_dir, 'params.npy'), params)
    compare_matlab_torch(data['B'][:, :, :, 0], params[:, :, :, 0], osp.join(work_dir, 'ktrans.pdf'))
    compare_matlab_torch(data['B'][:, :, :, 1], params[:, :, :, 1], osp.join(work_dir, 'kep.pdf'))
    compare_matlab_torch(data['B'][:, :, :, 2], params[:, :, :, 2], osp.join(work_dir, 't0.pdf'))

    # save results to dicom
    save_slices_to_dicom(data['B'][:, :, :, 0], dicom_dir=osp.join(work_dir, 'dicom/matlab-ktrans/'), SeriesDescription='Matlab-Ktrans')
    save_slices_to_dicom(data['B'][:, :, :, 1], dicom_dir=osp.join(work_dir, 'dicom/matlab-kep/'), SeriesDescription='Matlab-Kep')
    save_slices_to_dicom(data['B'][:, :, :, 2], dicom_dir=osp.join(work_dir, 'dicom/matlab-t0/'), SeriesDescription='Matlab-T0')
    #
    save_slices_to_dicom(params[:, :, :, 0], dicom_dir=osp.join(work_dir, 'dicom/torch-ktrans/'), SeriesDescription='Torch-Ktrans')
    save_slices_to_dicom(params[:, :, :, 1], dicom_dir=osp.join(work_dir, 'dicom/torch-kep/'), SeriesDescription='Torch-Kep')
    save_slices_to_dicom(params[:, :, :, 2], dicom_dir=osp.join(work_dir, 'dicom/torch-t0/'), SeriesDescription='Torch-T0')
