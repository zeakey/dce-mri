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

import time, sys, os
import os.path as osp
sys.path.insert(0, '/data1/Code/vlkit/vlkit/medical')
sys.path.insert(0, '..')

from utils import write_dicom, write_dicom, inference, save_slices_to_dicom, str2int, spatial_loss
from aif import get_aif, dispersed_aif
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



if __name__ == '__main__':
    np.random.seed(0)
    mmcv.runner.utils.set_random_seed(0)
    device = torch.device('cuda')

    dce_data = load_dce.load_dce_data('../dicom/10042_1_003Tnq2B/20180212/t1_twist_tra_dyn_29/', device=device)
    t2 = read_dicom_array('../dicom/10042_1_003Tnq2B/20180212/t2_tse_tra_320_p2_12/')
    data = loadmat('../tmp/parker_aif/10042_1_003Tnq2B-20180212.mat')
    h, w, c, _ = data['dce_ct'].shape
    weinmann_aif, aif_t = get_aif(aif='weinmann', max_base=6, acquisition_time=dce_data['acquisition_time'], device=device)
    parker_aif, _ = get_aif(aif='parker', max_base=6, acquisition_time=dce_data['acquisition_time'], device=device)

    x_tl, y_tl, x_br, y_br = 53, 57, 107, 120
    mask = torch.zeros(h, w, c, dtype=bool, device=device)
    z_mask = torch.zeros(c, dtype=bool, device=device)
    z_mask[10] = True
    mask[y_tl:y_br, x_tl:x_br, z_mask] = 1
    plt.imshow(mask[:, :, 10].cpu())
    plt.close()

    work_dir = 'work_dirs/AIF/parker-noise-scale/'

    cfg = Config.fromfile(osp.join(work_dir, 'noise_scale.py'))
    model = MODELS.build(cfg.model).to(device)
    model.load_state_dict(torch.load(osp.join(work_dir, 'model-iter50000.pth')))

    matlab_ktrans = data['ktrans']
    matlab_kep = data['kep']
    matlab_t0 = data['t0']

    ct = dce_data['ct'].to(device)

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
    t0_init = t0_init.cpu()

    torch.cuda.empty_cache()

    ktrans = ktrans_init.clone().to(device=device)
    kep = kep_init.clone().to(device=device)
    t0 = t0_init.clone().to(device=device)
    beta = torch.ones_like(ktrans).fill_(5e-2)

    ktrans.requires_grad = True
    kep.requires_grad = True
    beta.requires_grad = True

    params = [ktrans, kep, beta]
    optimizer = torch.optim.RMSprop(params=params, lr=1e-3, momentum=0)
    max_iters = 20
    scheduler = CosineScheduler(epoch_iters=max_iters, epochs=1)

    for i in range(max_iters):
        beta.clamp(5e-2, 0.5)
        aif_cp = dispersed_aif(parker_aif, aif_t, beta)
        # aif_cp = parker_aif
        ct1 = evaluate_curve(ktrans, kep, t0, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=aif_cp)
        l = torch.nn.functional.l1_loss(ct1, ct, reduction='none').sum(dim=-1).mean()
        # sl = spatial_loss(ktrans, uncertainty=uncertainty)
        # sl = ((noise_scale > 0.5) * sl).mean()
        sl = 0

        (l+sl*100).backward()
        optimizer.step()
        optimizer.zero_grad()
        print(l.item())
    print(aif_cp.shape)

    curve_iter = evaluate_curve(ktrans, kep, t0, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=aif_cp)

    [ct1, ktrans, kep, t0, curve_init, curve_iter] = map(remove_grad, [ct1, ktrans, kep, t0, curve_init, curve_iter])

    torch.cuda.empty_cache()

    ktrans = ktrans.cpu().clone()
    kep = kep.cpu().clone()
    t0 = t0.cpu().clone()


    ct = ct.cpu()
    ktrans_init = ktrans_init.cpu()
    kep_init = kep_init.cpu()
    t0_init = t0_init.cpu()
    curve_init = curve_init.cpu()
    loss_init = loss_init.cpu()
    #
    ktrans = ktrans.cpu()
    kep = kep.cpu()
    t0 = t0.cpu()
    loss_iter = loss_iter.cpu()
    curve_iter = curve_iter.cpu()
    aif_cp = aif_cp.detach().cpu()


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
    z = torch.zeros(positions.shape[0], dtype=torch.int).fill_(10)
    x, y = positions.split(dim=1, split_size=1)
    x = x.flatten()
    y = y.flatten()

    ncol = 7
    n = positions.shape[0]
    fig, axes = plt.subplots(n, ncol, figsize=(3*ncol, 3*n))

    for idx, i in enumerate(range(x.numel())):
        y1, x1, z1 = y[i].item(), x[i].item(), z[i].item()
        ct1 = ct[y1, x1, z1]

        params_init = torch.tensor([ktrans_init[y1, x1, z1].item(), kep_init[y1, x1, z1].item(), t0_init[y1, x1, z1].item()])
        params_iter = torch.tensor([ktrans[y1, x1, z1].item(), kep_init[y1, x1, z1].item(), t0_init[y1, x1, z1].item()])

        j = 0
        axes[idx, j].plot(ct1)
        axes[idx, j].plot(curve_init[y1, x1, z1])
        axes[idx, j].set_title('Transfomer param:\n %.3f %.3f %.3f \n loss=%.3f'  % (params_init[0], params_init[1], params_init[2], loss_init[y1, x1, z1]))

        j += 1
        axes[idx, j].plot(ct1)
        axes[idx, j].plot(curve_iter[y1, x1, z1])
        axes[idx, j].set_title('Iter param:\n %.3f %.3f %.3f \n loss=%.3f'  % (params_iter[0], params_iter[1], params_iter[2], loss_iter[y1, x1, z1]))


        j += 1
        axes[idx, j].plot(aif_cp[y1, x1, z1])
        axes[idx, j].set_title('AIF: $\\beta$=%.3f'  % beta[y1, x1, z1])

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
        axes[idx, j].imshow(norm01(loss_init[:, :, z1].numpy())[y_tl:y_br, x_tl:x_br])
        axes[idx, j].scatter(x1-x_tl, y1-y_tl, marker='x', color='red')
        axes[idx, j].set_title('loss_init %.3f' % loss_init[y1, x1, z1])

    plt.tight_layout(h_pad=3)
    plt.savefig('relative-loss-10042_1_003Tnq2B.pdf')
    plt.close()