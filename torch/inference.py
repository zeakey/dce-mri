import matplotlib
matplotlib.use('agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import vlkit.plt as vlplt

from scipy.stats import beta

import torch
from torch.distributions import Beta
import numpy as np
from mmcv.cnn import MODELS
from mmcv.utils import Config

from tqdm import tqdm
from einops import rearrange

import mmcv
from vlkit.image import norm01, norm255
from vlkit.medical.dicomio import read_dicom_array
from vlkit.lrscheduler import CosineScheduler

from scipy.io import loadmat, savemat
import pydicom

import time, sys, os, shutil
import os.path as osp
sys.path.insert(0, '/data1/Code/vlkit/vlkit/medical')
sys.path.insert(0, '..')

from utils import write_dicom, write_dicom, inference, save_slices_to_dicom, str2int, spatial_loss
from aif import get_aif, dispersed_aif, interp_aif
from pharmacokinetic import fit_slice, process_patient, np2torch, evaluate_curve
import load_dce
from models import DCETransformer

from pharmacokinetic import calculate_reconstruction_loss



def remove_grad(x):
    x = x.detach()
    x.requires_grad = False
    x.grad = None
    return x


def rel_loss(x, y):
    l = torch.nn.functional.l1_loss(x, y, reduction='none').sum(dim=-1)
    s = torch.maximum(x, y).sum(dim=-1).clamp(min=1e-6)
    print('l: ', l.min(), l.max(), 's: ', s.min(), s.max(), 'div: ', (l / s).min(), (l / s).max())
    return (l / s).mean()


def save2dicom(array, save_dir, example_dicom, description, pad=False, **kwargs):
    if pad:
        pad = np.zeros((array.shape[0], array.shape[1], 3), dtype=array.dtype)
        array = np.concatenate((pad, array, pad), axis=-1)
    SeriesNumber = int(str(str2int(description))[-12:])
    SeriesInstanceUID = str(SeriesNumber)
    save_dir = osp.join(save_dir, description)
    save_slices_to_dicom(
        array,
        dicom_dir=save_dir,
        example_dicom=example_dicom,
        SeriesNumber=SeriesNumber,
        SeriesInstanceUID=SeriesInstanceUID,
        SeriesDescription=description,
        **kwargs)


def process_patient(dce_dir, save_dir='results/dispersion'):

    cad_ktrans_dir = load_dce.find_ktrans_folder(osp.join(dce_dir, '../'))
    t2_dir = load_dce.find_t2_folder(osp.join(dce_dir, '../'))

    dce_data = load_dce.load_dce_data(dce_dir, device=device)

    weinmann_aif, aif_t = get_aif(aif='weinmann', max_base=6, acquisition_time=dce_data['acquisition_time'], device=device)
    parker_aif, _ = get_aif(aif='parker', max_base=6, acquisition_time=dce_data['acquisition_time'], device=device)
    weinmann_aif *= parker_aif.sum() / weinmann_aif.sum()

    work_dir = 'work_dirs/AIF/parker-noise-scale/'
    cfg = Config.fromfile(osp.join(work_dir, 'noise_scale.py'))
    model = MODELS.build(cfg.model).to(device)
    model.load_state_dict(torch.load(osp.join(work_dir, 'model-iter50000.pth')))

    ct = dce_data['ct'].to(device)[:, :, 3:17, :]

    torch.cuda.empty_cache()
    tic = time.time()
    output = inference(model, ct)
    ktrans_init, kep_init, t0_init, noise_scale = output.split(dim=-1, split_size=1)

    del model

    ktrans_init = ktrans_init.squeeze(-1)
    kep_init = kep_init.squeeze(-1)
    t0_init = t0_init.squeeze(-1)
    noise_scale = noise_scale.squeeze(-1)
    uncertainty = noise_scale.neg().multiply(1).exp()

    toc = time.time()
    print(toc-tic, 'seconds')

    curve_init = evaluate_curve(ktrans_init, kep_init, t0_init, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=parker_aif).cpu()
    loss_init = calculate_reconstruction_loss(
        ktrans_init,
        kep_init,
        t0_init,
        ct,
        t=dce_data['acquisition_time'],
        aif_t=aif_t,
        aif_cp=parker_aif
    ).cpu()
    ktrans_init = ktrans_init.cpu()
    kep_init = kep_init.cpu()

    torch.cuda.empty_cache()
    max_iters = 20
    dtype = torch.float32

    # -------------------------------------------------------------------------------------------- #
    ktrans_iter = ktrans_init.clone().to(device=device, dtype=dtype).requires_grad_()
    kep_iter = kep_init.clone().to(ktrans_iter).requires_grad_()
    parker_aif = parker_aif.to(ktrans_iter)
    aif_t = aif_t.to(ktrans_iter)
    ct = ct.to(ktrans_iter)

    params = [ktrans_iter, kep_iter]
    optimizer = torch.optim.RMSprop(params=params, lr=1e-3)

    scheduler = CosineScheduler(epoch_iters=max_iters, epochs=1, warmup_iters=max_iters//5, max_lr=1e-3, min_lr=1e-5)

    for i in range(max_iters):
        ct_iter = evaluate_curve(ktrans_iter, kep_iter, t0_init, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=parker_aif)
        loss_iter = torch.nn.functional.l1_loss(ct_iter, ct, reduction='none').sum(dim=-1)
        l = loss_iter.mean()
        # sl = spatial_loss(ktrans, uncertainty=uncertainty)
        # sl = ((noise_scale > 0.5) * sl).mean()
        sl = 0

        lr = scheduler.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        (l+sl*100).backward()
        optimizer.step()
        optimizer.zero_grad()
        print("[{i:03d}/{max_iters:03d}] lr={lr:.2e} loss={loss:.4f}".format(lr=lr, loss=l.item(), i=i+1, max_iters=max_iters))

    ct_iter = ct_iter.detach().cpu()
    ktrans_iter = ktrans_iter.detach().cpu()
    kep_iter = kep_iter.detach().cpu()
    loss_iter = loss_iter.detach().cpu()

    aif_t = aif_t.to(device=device, dtype=dtype)
    ct = ct.to(aif_t)

    # -------------------------------------------------------------------------------------------- #
    # dispersed AIFs
    # -------------------------------------------------------------------------------------------- #
    beta_dsps = []
    ktrans_dsps = []
    kep_dsps = []
    aif_dsps = []
    loss_dsps = []
    ct_dsps = []

    num_betas = 20
    betas = torch.linspace(5e-2, 5e-1, num_betas)

    for idx, b in enumerate(betas):
        torch.cuda.empty_cache()
        ktrans_dsps1 = ktrans_init.clone().to(aif_t).requires_grad_()
        kep_dsps1 = kep_init.clone().to(ktrans_dsps1).requires_grad_()
        beta_dsps1 = torch.ones_like(ktrans_dsps1).fill_(b).requires_grad_()
        parker_aif = parker_aif.to(ktrans_dsps1)

        params = [ktrans_dsps1, kep_dsps1, beta_dsps1]
        optimizer = torch.optim.RMSprop(params=params, lr=1e-3)
        scheduler = CosineScheduler(epoch_iters=max_iters, epochs=1, warmup_iters=max_iters//5, max_lr=1e-3, min_lr=1e-5)

        for i in range(max_iters):
            beta_dsps1.clamp(5e-2, 5e-1)
            aif_dsps1 = dispersed_aif(parker_aif, aif_t, beta_dsps1)
            ct_dsps1 = evaluate_curve(ktrans_dsps1, kep_dsps1, t0_init, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=aif_dsps1)
            loss_dsps1 = torch.nn.functional.l1_loss(ct_dsps1, ct, reduction='none').sum(dim=-1)
            l = loss_dsps1.mean()
            # sl = spatial_loss(ktrans, uncertainty=uncertainty)
            # sl = ((noise_scale > 0.5) * sl).mean()
            sl = 0

            lr = scheduler.step()
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            (l+sl*100).backward()
            optimizer.step()
            optimizer.zero_grad()
            print("[{idx:02d}/{num_betas:02d}][{i:03d}/{max_iters:03d}] lr={lr:.2e} loss={loss:.4f}".format(
                idx=idx+1,
                num_betas=num_betas,
                lr=lr, loss=l.item(),
                max_iters=max_iters,
                i=i+1))

        ct_dsps.append(ct_dsps1.detach().cpu())
        ktrans_dsps.append(ktrans_dsps1.detach().cpu())
        kep_dsps.append(kep_dsps1.detach().cpu())
        beta_dsps.append(beta_dsps1.detach().cpu())
        aif_dsps.append(aif_dsps1.detach().cpu())
        loss_dsps.append(loss_dsps1.detach().cpu())

    # pick values with minimal loss
    loss_dsps = torch.stack(loss_dsps, dim=-1)
    ktrans_dsps = torch.stack(ktrans_dsps, dim=-1)
    kep_dsps = torch.stack(kep_dsps, dim=-1)
    beta_dsps = torch.stack(beta_dsps, dim=-1)
    ct_dsps = torch.stack(ct_dsps, dim=-1)
    aif_dsps = torch.stack(aif_dsps, dim=-1)

    loss_dsps, min_loss_dsps_indices = loss_dsps.min(dim=-1)
    min_loss_dsps_indices = min_loss_dsps_indices.to(torch.int64).unsqueeze(dim=-1)

    beta_dsps = torch.gather(beta_dsps, dim=-1, index=min_loss_dsps_indices).squeeze(dim=-1)
    ktrans_dsps = torch.gather(ktrans_dsps, dim=-1, index=min_loss_dsps_indices).squeeze(dim=-1)
    kep_dsps = torch.gather(kep_dsps, dim=-1, index=min_loss_dsps_indices).squeeze(dim=-1)
    ct_dsps = torch.gather(ct_dsps, dim=-1, index=min_loss_dsps_indices.repeat(1, 1, 1, ct_dsps.shape[-2]).unsqueeze(dim=-1)).squeeze(dim=-1)
    aif_dsps = torch.gather(aif_dsps, dim=-1, index=min_loss_dsps_indices.repeat(1, 1, 1, aif_dsps.shape[-2]).unsqueeze(dim=-1)).squeeze(dim=-1)

    # -------------------------------------------------------------------------------------------- #
    torch.cuda.empty_cache()
    ktrans_interp_aif = ktrans_init.clone().to(device=device, dtype=dtype).requires_grad_()
    kep_interp_aif = kep_init.clone().to(ktrans_interp_aif).requires_grad_()
    beta_interp_aif = torch.ones_like(ktrans_interp_aif).requires_grad_()
    parker_aif = parker_aif.to(ktrans_interp_aif)
    aif_t = aif_t.to(ktrans_interp_aif)
    ct = ct.to(ktrans_interp_aif)

    params = [ktrans_interp_aif, kep_interp_aif, beta_interp_aif]
    optimizer = torch.optim.RMSprop(params=params, lr=1e-3)
    # optimizer = torch.optim.SGD(params=params, lr=1e-4, momentum=0)
    scheduler = CosineScheduler(epoch_iters=max_iters, epochs=1, warmup_iters=max_iters//5, max_lr=1e-3, min_lr=1e-5)

    for i in range(max_iters):
        beta_interp_aif.clamp(0, 1)
        aif_interp_aif = interp_aif(parker_aif, weinmann_aif, beta_interp_aif)
        # aif_cp = parker_aif
        ct_interp_aif = evaluate_curve(ktrans_interp_aif, kep_interp_aif, t0_init, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=aif_interp_aif)
        loss_interp_aif= torch.nn.functional.l1_loss(ct_interp_aif, ct, reduction='none').sum(dim=-1)
        l = loss_interp_aif.mean()
        # sl = spatial_loss(ktrans, uncertainty=uncertainty)
        # sl = ((noise_scale > 0.5) * sl).mean()
        sl = 0

        lr = scheduler.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        (l+sl*100).backward()
        optimizer.step()
        optimizer.zero_grad()
        print("[{i:03d}/{max_iters:03d}] lr={lr:.2e} loss={loss:.4f}".format(lr=lr, loss=l.item(), i=i+1, max_iters=max_iters))

    ct_interp_aif = ct_interp_aif.detach().cpu()
    ktrans_interp_aif = ktrans_interp_aif.detach().cpu()
    kep_interp_aif = kep_interp_aif.detach().cpu()
    beta_interp_aif = beta_interp_aif.detach().cpu()
    aif_interp_aif = aif_interp_aif.detach().cpu()
    loss_interp_aif = loss_interp_aif.detach().cpu()

    torch.cuda.empty_cache()

    ct = ct.cpu()
    ktrans_init = ktrans_init.cpu()
    kep_init = kep_init.cpu()
    t0_init = t0_init.cpu()
    curve_init = curve_init.cpu()
    loss_init = loss_init.cpu()

    # example_dicom = '/media/hdd1/IDX_Current/dicom/10042_1_004D6Sy8/20160616/iCAD-Ktrans-FA-0-E_30009'
    example_dicom = t2_dir
    shared_kwargs = dict(PatientID=dce_data['patient'].upper(), PatientName=dce_data['patient'].upper(), example_dicom=example_dicom)

    save_dir1 = osp.join(save_dir, dce_data['patient'])
    shutil.copytree(cad_ktrans_dir, osp.join(save_dir1, cad_ktrans_dir.split(os.sep)[-1]))
    save2dicom(beta_dsps.cpu().numpy(), osp.join(save_dir, dce_data['patient']), description='dsps-beta', pad=True, **shared_kwargs)
    save2dicom(ktrans_dsps.cpu().numpy(), osp.join(save_dir, dce_data['patient']), description='dsps-ktrans', pad=True, **shared_kwargs)


if __name__ == '__main__':
    np.random.seed(0)
    mmcv.runner.utils.set_random_seed(0)
    device = torch.device('cuda')

    for dce_dir in load_dce.find_dce_folders('../dicom/'):
        # data = loadmat('../tmp/parker_aif/10042_1_003Tnq2B-20180212.mat')
        process_patient(dce_dir, save_dir='results/dispersion-1')

    sys.exit()

    positions = torch.tensor([
        [80, 80],
        [69, 110],
        [68, 108],
        [68, 109],
        [68, 110],
        [69, 108],
        [69, 109],
        [69, 110],
        #
        [104, 90],
        [104, 88],
        [104, 89],
        [104, 91],
        #
        [42+x_tl, 48+y_tl],
    ])

    mask = torch.zeros(160, 160)
    mask[y_tl:y_br, x_tl:x_br] = True
    y, x = torch.where(mask)

    n = 150
    inds = np.random.choice(x.numel(), min(n, x.numel()))
    x = x[inds].view(-1, 1)
    y = y[inds].view(-1, 1)
    positions = torch.cat((positions, torch.cat((y, x), dim=1)), dim=0) 

    z = torch.zeros(positions.shape[0], dtype=torch.int).fill_(5)
    x, y = positions.split(dim=1, split_size=1)
    x = x.flatten()
    y = y.flatten()


    ncol = 16
    n = positions.shape[0]
    fig, axes = plt.subplots(n, ncol, figsize=(3*ncol, 3*n))

    for idx, i in enumerate(range(x.numel())):
        y1, x1, z1 = y[i].item(), x[i].item(), z[i].item()
        ct1 = ct[y1, x1, z1]

        params_init = torch.tensor([ktrans_init[y1, x1, z1].item(), kep_init[y1, x1, z1].item(), t0_init[y1, x1, z1].item()])
        params_iter = torch.tensor([ktrans_iter[y1, x1, z1].item(), kep_iter[y1, x1, z1].item(), t0_init[y1, x1, z1].item()])
        params_dsps = torch.tensor([ktrans_dsps[y1, x1, z1].item(), kep_dsps[y1, x1, z1].item(), t0_init[y1, x1, z1].item()])
        params_interp_aif = torch.tensor([ktrans_interp_aif[y1, x1, z1].item(), kep_interp_aif[y1, x1, z1].item(), t0_init[y1, x1, z1].item()])

        j = 0
        axes[idx, j].plot(ct1)
        axes[idx, j].plot(curve_init[y1, x1, z1])
        axes[idx, j].set_title('Transfomer param:\n %.3f %.3f %.3f \n loss=%.3f'  % (params_init[0], params_init[1], params_init[2], loss_init[y1, x1, z1]))

        j += 1
        axes[idx, j].plot(ct1)
        axes[idx, j].plot(ct_iter[y1, x1, z1])
        axes[idx, j].set_title('Iter param:\n %.3f %.3f %.3f \n loss=%.3f'  % (params_iter[0], params_iter[1], params_iter[2], loss_iter[y1, x1, z1]))

        j += 1
        axes[idx, j].plot(ct1)
        axes[idx, j].plot(ct_dsps[y1, x1, z1])
        axes[idx, j].set_title('Iter param (disperse):\n %.3f %.3f %.3f \n loss=%.3f'  % (params_dsps[0], params_dsps[1], params_dsps[2], loss_dsps[y1, x1, z1]))

        j += 1
        axes[idx, j].plot(ct1)
        axes[idx, j].plot(ct_interp_aif[y1, x1, z1])
        axes[idx, j].set_title('Iter param (interp_aif):\n %.3f %.3f %.3f \n loss=%.3f'  % (params_interp_aif[0], params_interp_aif[1], params_interp_aif[2], loss_interp_aif[y1, x1, z1]))

        j += 1
        axes[idx, j].plot(aif_dsps[y1, x1, z1])
        axes[idx, j].set_title('AIF (dispersed): $\\beta$=%.3f'  % beta_dsps[y1, x1, z1])
        axes[idx, j].set_ylim(0, 10.5)
        axes[idx, j].grid(True)

        j += 1
        axes[idx, j].plot(aif_interp_aif[y1, x1, z1])
        axes[idx, j].set_title('AIF (interp): $\\beta$=%.3f'  % beta_interp_aif[y1, x1, z1])
        axes[idx, j].set_ylim(0, 10.5)
        axes[idx, j].grid(True)

        j = j + 1
        t2im = mmcv.imresize(norm01(t2[z1]), (h, w))
        axes[idx, j].imshow(t2im)
        rect = patches.Rectangle((x_tl, y_tl), x_br-x_tl, y_br-y_tl, linewidth=1, edgecolor='black', facecolor='none')
        axes[idx, j].add_patch(rect)
        axes[idx, j].set_title('T2 (full) \n (%d, %d, %d)' % (x1, y1, z1))
        axes[idx, j].scatter(x1, y1, marker='x', color='red')

        j = j + 1
        axes[idx, j].imshow(t2im[y_tl:y_br, x_tl:x_br])
        axes[idx, j].set_title('T2 (ROI) \n (%d, %d, %d)' % (x1, y1, z1))

        j = j + 1
        axes[idx, j].imshow(norm01(ktrans_init[:, :, z1].numpy())[y_tl:y_br, x_tl:x_br])
        axes[idx, j].scatter(x1-x_tl, y1-y_tl, marker='x', color='red')
        axes[idx, j].set_title('Ktrans init')

        j = j + 1
        axes[idx, j].imshow(norm01(ktrans_iter[:, :, z1].numpy())[y_tl:y_br, x_tl:x_br])
        axes[idx, j].scatter(x1-x_tl, y1-y_tl, marker='x', color='red')
        axes[idx, j].set_title('Ktrans iter')

        j = j + 1
        axes[idx, j].imshow(norm01(beta_dsps[:, :, z1].numpy())[y_tl:y_br, x_tl:x_br])
        axes[idx, j].scatter(x1-x_tl, y1-y_tl, marker='x', color='red')
        axes[idx, j].set_title('$\\beta$ (dsps)')

        j = j + 1
        axes[idx, j].imshow(norm01(beta_interp_aif[:, :, z1].numpy())[y_tl:y_br, x_tl:x_br])
        axes[idx, j].scatter(x1-x_tl, y1-y_tl, marker='x', color='red')
        axes[idx, j].set_title('$\\beta$ (interp)')

        j = j + 1
        axes[idx, j].imshow(norm01(loss_init[:, :, z1].numpy())[y_tl:y_br, x_tl:x_br])
        axes[idx, j].scatter(x1-x_tl, y1-y_tl, marker='x', color='red')
        axes[idx, j].set_title('loss_init %.3f' % loss_init[y1, x1, z1])

        j = j + 1
        axes[idx, j].imshow(norm01(loss_iter[:, :, z1].numpy())[y_tl:y_br, x_tl:x_br])
        axes[idx, j].scatter(x1-x_tl, y1-y_tl, marker='x', color='red')
        axes[idx, j].set_title('loss_iter %.3f' % loss_iter[y1, x1, z1])

        j = j + 1
        axes[idx, j].imshow(norm01(loss_dsps[:, :, z1].numpy())[y_tl:y_br, x_tl:x_br])
        axes[idx, j].scatter(x1-x_tl, y1-y_tl, marker='x', color='red')
        axes[idx, j].set_title('loss_dsps %.3f' % loss_dsps[y1, x1, z1])

        j = j + 1
        axes[idx, j].imshow(norm01(loss_interp_aif[:, :, z1].numpy())[y_tl:y_br, x_tl:x_br])
        axes[idx, j].scatter(x1-x_tl, y1-y_tl, marker='x', color='red')
        axes[idx, j].set_title('loss_interp_aif %.3f' % loss_interp_aif[y1, x1, z1])

    plt.tight_layout(h_pad=3)
    plt.savefig('relative-loss-10042_1_003Tnq2B.pdf')
    plt.close()