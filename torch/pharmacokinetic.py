from multiprocessing import reduction
import torch, math, os, pydicom, warnings
import os.path as osp
import numpy as np
from einops import rearrange
from vlkit.lrscheduler import CosineScheduler, MultiStepScheduler

from tofts import tofts, tofts1d, tofts3d


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


def evaluate_curve(ktrans, kep, t0, aif_t, aif_cp, t):
    assert type(ktrans) == type(kep) == type(t0)
    assert ktrans.numel() == kep.numel() == t0.numel()

    if isinstance(ktrans, float):
        ktrans = torch.tensor(ktrans).to(aif_t)
        kep = torch.tensor(kep).to(aif_t)
        t0 = torch.tensor(t0).to(aif_t)

    if ktrans.numel() == 1:
        ktrans = ktrans.view(1, 1)
        kep = kep.view(1, 1)
        t0 = t0.view(1, 1)
    assert ktrans.shape == kep.shape == t0.shape

    # unify device
    kep = kep.to(ktrans)
    t0 = t0.to(ktrans)
    aif_t = aif_t.to(ktrans)
    aif_cp = aif_cp.to(ktrans)
    t = t.to(ktrans)

    orig_shape = list(ktrans.shape)
    n = ktrans.numel()
    if aif_cp.ndim==1 or (aif_cp.ndim == 2 and aif_cp.size(0) == 1):
        aif_cp = aif_cp.view(1, -1).repeat(n, 1)
    elif aif_cp.ndim == 2:
        pass
    elif aif_cp.ndim == len(orig_shape) + 1:
        aif_cp = rearrange(aif_cp, 'h w s d -> (h w s) d')
    else:
        raise ValueError(f"invalid shape {aif_cp.shape}")
    ct1 = tofts1d(ktrans.view(n), kep.view(n), t0.view(n), t.view(1, -1).repeat(n, 1), aif_t.view(1, -1).repeat(n, 1), aif_cp)
    return ct1.view(orig_shape + [-1])

    # AIFs
    if aif_cp.ndim == 1:
        aif_cp = aif_cp.view([1] * len(orig_shape) + [aif_cp.shape[0]])
        aif_cp = aif_cp.repeat(orig_shape + [1])
    else:
        assert list(aif_cp.shape)[:-1] == list(ktrans.shape)

    if aif_t.ndim == 1:
        aif_t = aif_t.view([1] * len(orig_shape) + [aif_t.shape[0]])
        aif_t = aif_t.repeat(orig_shape + [1])
    else:
        assert list(aif_t.shape)[:-1] == list(ktrans.shape)

    ktrans = ktrans.reshape(-1, 1)
    kep = kep.reshape(-1, 1)
    t0 = t0.reshape(-1, 1)
    aif_t = aif_t.reshape(-1, 1, aif_t.shape[-1])
    aif_cp = aif_cp.reshape(-1, 1, aif_cp.shape[-1])

    n = ktrans.shape[0]
    t = t.view(1, 1, -1).repeat(n, 1, 1)

    ct = tofts(ktrans, kep, t0, t, aif_t, aif_cp)
    assert torch.all(ct.squeeze(dim=1) == ct1)
    signal_len = ct.shape[-1]
    output_shape = orig_shape + [signal_len]
    return ct.view(output_shape)


def calculate_reconstruction_loss(ktrans, kep, t0, ct, t, aif_t, aif_cp, relative=False):
    assert ktrans.shape == kep.shape == t0.shape
    ct_recon = evaluate_curve(ktrans, kep, t0, aif_t=aif_t, aif_cp=aif_cp, t=t)
    loss = torch.nn.functional.l1_loss(ct_recon, ct, reduction='none').sum(dim=-1)
    if relative:
        ct_sum = ct.sum(dim=-1)
        loss = loss / ct_sum
        loss[ct.mean(dim=-1) <= 1e-2] = 0
    return loss


def find_max_base(ct):
    assert ct.ndim == 4
    h, w, slices, frames = ct.shape

    x1, x2 = -21 + w // 2, 20 + w // 2
    y1, y2 = -21 + h // 2, 20 + h // 2
    center = ct[y1:y2, x1:x2, slices//2-1, :].mean(dim=(0, 1))
    grad = torch.gradient(center)[0]
    max_grad_idx = grad.argmax()
    neg_grad_idx = torch.where(grad[:max_grad_idx] < 0)[0]
    bnum = neg_grad_idx.max() + 1
    baseline = center[:bnum+1].mean()

    while center[bnum+1] / baseline < 1.01:
        bnum += 1
        baseline = center[:bnum+1].mean()
    return bnum, center, grad


def gd_concentration(dce, max_base, t10=None, tr=3.89, fa=12):
    assert dce.ndim == 4
    fa = 2 * math.pi * fa / 360
    rr1 = 3.9 # t1 relaxivity at 3T
    h, w, slices, frames = dce.shape
    if t10 is None:
        t10 = torch.ones((h, w, slices), device=dce.device) * 1998
    # E10 = exp(-tr/t10);
    e10 = (-tr / t10).exp()
    # B = (1-E10)/(1-cosd(fa)*E10);
    b = (1 - e10) / (1 - math.cos(fa) * e10)
    r10 = (1000.0 / t10).unsqueeze(dim=-1).expand_as(dce)

    mask = torch.zeros((h, w, slices), device=dce.device)
    threshold = dce[:, :, :, -1].reshape(-1, slices).max(dim=0).values.view(1, 1, -1) * 0.05
    mask = dce[:, :, :, -1] > threshold

    baseline = dce[:, :, :, :max_base+1].mean(dim=-1, keepdim=True)
    # baseline[mask] = 1

    enhanced = dce / baseline.expand_as(dce)
    b_expand = 0.8 / b.unsqueeze(dim=-1).expand_as(enhanced)
    enhanced = torch.where(enhanced > b_expand, b_expand, enhanced)
    a = b.unsqueeze(dim=-1) * enhanced
    r1 = (-1000.0 / tr) * ((1 - a) / (1 - math.cos(fa) * a)).log()
    concentration = torch.where(r1 > r10, (r1 - r10) / rr1, torch.zeros_like(dce))
    concentration[:, :, :, :max_base+1] = 0
    concentration[mask.logical_not()] = 0
    return concentration


def compare_results(param1, param2, name1='name1', name2='name2', fig_filename='results.pdf'):
    slices = param1.shape[2]
    assert param1.shape[2] == param2.shape[2]
    os.makedirs(osp.dirname(fig_filename), exist_ok=True)

    fig, axes = plt.subplots(slices, 2, figsize=(4, 20))
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_title(name1)
    axes[0, 1].set_title(name2)
    for i in range(0, slices):
        axes[i, 0].imshow(param1[:, :, i])
        axes[i, 1].imshow(param2[:, :, i])
        axes[i, 0].set_ylabel("slice#%.2d"%i)
    plt.tight_layout()
    plt.savefig(fig_filename)
    plt.close()



def np2torch(data, device=torch.device('cpu')):
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            if v.dtype != np.float32:
                try:
                    v = v.astype(np.float32)
                except:
                    pass
            try:
                data[k] = torch.tensor(v).float().to(device=device)
            except:
                pass
    return data


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from scipy.io import loadmat, savemat
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import time, sys
    from load_dce import read_dce_dicoms

    dicom_data = read_dce_dicoms('../dicom/10042_1_1Wnr0444/20161209/iCAD-MCC_33000')
    dce_data = torch.tensor(dicom_data['data'].astype(np.float32))
    acquisition_time = torch.tensor(dicom_data['acquisition_time'])
    repetition_time = torch.tensor(dicom_data['repetition_time'])
    flip_angle = torch.tensor(dicom_data['flip_angle'])

    max_base = find_max_base(dce_data)

    # important! shift max_base for uniform ct
    dce_data = dce_data[:, :, :, max_base:]
    dce_data = torch.cat((
            dce_data,
            dce_data[:, :, :, -1:].repeat(1, 1, 1, max_base)
        ), dim=-1)

    acquisition_time = acquisition_time[max_base:]
    interval = (acquisition_time[-1] - acquisition_time[0]) / (acquisition_time.numel() - 1)

    acquisition_time = torch.cat((acquisition_time, acquisition_time[-1] + torch.arange(1, max_base+1) * interval), dim=0)
    acquisition_time = acquisition_time - acquisition_time[0]
    # second to minite
    acquisition_time = acquisition_time / 60

    max_base = 0

    ct = gd_concentration(dce_data, max_base=max_base)

    acquisition_time = acquisition_time / 60
    # supporse maximal aif_time is 7 minutes
    aif_time = torch.arange(0, 7, 1/60, dtype=torch.float64).to(acquisition_time)

    # dce_data1 = load_dce('../dicom/10042_1_1Wnr0444/20161209/iCAD-MCC_33000')
    # dce_data2 = load_dce('../dicom/10042_1_3L4Hhlz5/20160919/iCAD-MCC_33000')
    # dce_data3 = load_dce('../dicom/10042_1_5Xb55H46/20170825/iCAD-MCC_33000')
    dce_data4 = load_dce('/data1/kzhao/prostate-mri-reorganised/10042_1_003Tnq2B/DCE')

    hct = 0.42
    aif_cp = parker_aif(
        a1=0.809, a2=0.330,
        t1=0.17046, t2=0.365,
        sigma1=0.0563, sigma2=0.132,
        alpha=1.050, beta=0.1685, s=38.078, tau=0.483,
        t=aif_time,
        ) / (1 - hct)

    # save_slices_to_dicom(np.random.randn(160, 160, 20), 'work_dirs/debug', SeriesDescription='PyTorch-Ktrans')
    # sys.exit()

    work_dir = 'work_dirs/RMSprop_lr1e-3~1e-8_iter500'

    device = torch.device('cuda')

    data = loadmat('../tmp/patient-0.mat')
    data = np2torch(data, device=device)

    time_dce = data['time_dce'] / 60
    aif_t = torch.arange(0, time_dce[-1].item(), 1/60, dtype=torch.float64).to(time_dce)

    max_base = find_max_base(data['dce'])

    ct = gd_concentration(data['dce'], max_base=max_base)

    hct = 0.42
    aif_cp = parker_aif(
        a1=0.809, a2=0.330,
        t1=0.17046, t2=0.365,
        sigma1=0.0563, sigma2=0.132,
        alpha=1.050, beta=0.1685, s=38.078, tau=0.483,
        t=aif_t - (time_dce[int(data['maxBase'].item())-1] * 60).ceil() / 60
        ) / (1 - hct)

    tic = time.time()
    param, loss = process_patient(
        data['time_dce'] / 60,
        data['dce_ct'], aif_t,
        aif_cp,
        max_iter=500,
        work_dir=work_dir
    )
    toc = time.time()
    print("Done, ETA=%.3f" % (toc-tic))
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
