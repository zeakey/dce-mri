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
from mmcv.utils import Config, DictAction, get_logger

from tqdm import tqdm
from einops import rearrange

import mmcv
from vlkit.image import norm01, norm255
from vlkit.medical.dicomio import read_dicoms
from vlkit.lrscheduler import CosineScheduler

from scipy.io import loadmat, savemat
import pydicom

import time, sys, os, shutil, argparse
import os.path as osp
sys.path.insert(0, '/data1/Code/vlkit/vlkit/medical')
sys.path.insert(0, '..')

from utils import write_dicom, write_dicom, inference, save_slices_to_dicom, str2int, spatial_loss
from aif import get_aif, dispersed_aif, interp_aif
from pharmacokinetic import fit_slice, process_patient, np2torch, evaluate_curve
from utils import load_dce, find_image_dirs
from models import DCETransformer
from pharmacokinetic import calculate_reconstruction_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint weights')
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


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

def save2dicom(array, save_dir, example_dicom, description, **kwargs):
    SeriesNumber = int(str(str2int(description))[-12:])
    SeriesInstanceUID = str(SeriesNumber)
    save_dir = osp.join(save_dir, description)
    save_slices_to_dicom(
        array,
        save_dir=save_dir,
        example_dicom=example_dicom,
        SeriesNumber=SeriesNumber,
        SeriesInstanceUID=SeriesInstanceUID,
        SeriesDescription=description,
        **kwargs)


def process_patient(cfg, dce_dir, save_dir='results/dispersion'):
    patient_id = dce_dir.split(os.sep)[-3]
    cad_ktrans_dir = find_image_dirs.find_ktrans_folder(osp.join(dce_dir, '../'))
    t2_dir = find_image_dirs.find_t2_folder(osp.join(dce_dir, '../'))

    if cad_ktrans_dir and t2_dir:
        t2w = read_dicoms(t2_dir)
        cad_ktrans = read_dicoms(cad_ktrans_dir)
    else:
        logger.info(f"{patient_id}: cannot find t2w or cad_ktrans, pass")
        return None
    try:
        dce_data = load_dce.load_dce_data(dce_dir, device=cfg.device)
    except:
        return None

    weinmann_aif, aif_t = get_aif(aif='weinmann', max_base=6, acquisition_time=dce_data['acquisition_time'], device=cfg.device)
    parker_aif, _ = get_aif(aif='parker', max_base=6, acquisition_time=dce_data['acquisition_time'], device=cfg.device)

    model = MODELS.build(cfg.model).to(cfg.device)
    model.load_state_dict(torch.load(cfg.checkpoint))

    ct = dce_data['ct'].to(cfg.device)

    torch.cuda.empty_cache()
    tic = time.time()
    output = inference(model, ct)
    del model

    if cfg.aif == 'mixed':
        ktrans_init, kep_init, t0_init, noise_scale, beta_init = output.values()
    else:
        ktrans_init, kep_init, t0_init, noise_scale = output.values()

    ktrans_init = ktrans_init.squeeze(-1)
    kep_init = kep_init.squeeze(-1)
    t0_init = t0_init.squeeze(-1)
    noise_scale = noise_scale.squeeze(-1)

    if cfg.aif == 'mixed':
        beta_init = beta_init.squeeze(-1)
        aif_cp = interp_aif(parker_aif, weinmann_aif, beta=beta_init)
    else:
        aif_cp = parker_aif

    toc = time.time()
    logger.info('%.2f seconds' % (toc-tic))

    curve_init = evaluate_curve(ktrans_init, kep_init, t0_init, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=aif_cp).cpu()
    loss_init = calculate_reconstruction_loss(
        ktrans_init,
        kep_init,
        t0_init,
        ct,
        t=dce_data['acquisition_time'],
        aif_t=aif_t,
        aif_cp=aif_cp
    ).cpu()
    ktrans_init = ktrans_init.cpu()
    kep_init = kep_init.cpu()

    torch.cuda.empty_cache()
    dtype = torch.float32

    # -------------------------------------------------------------------------------------------- #
    ktrans_iter = ktrans_init.clone().to(device=cfg.device, dtype=dtype).requires_grad_()
    kep_iter = kep_init.clone().to(ktrans_iter).requires_grad_()
    beta_iter = torch.ones_like(kep_iter).requires_grad_()
    # if cfg.aif == 'mixed':
    #     beta_iter = beta_init.clone().to(ktrans_iter).requires_grad_()

    aif_t = aif_t.to(ktrans_iter)
    ct = ct.to(ktrans_iter)
    # optimizer = torch.optim.AdamW(params=[{'params': [ktrans_iter, kep_iter], 'lr': 1e-1}, {'params': [beta_iter], 'lr': 1e-3}])
    optimizer = torch.optim.RMSprop(params=[{'params': [ktrans_iter, kep_iter], 'lr': 1e-1}, {'params': [beta_iter], 'lr': 1e-3}])
    scheduler = CosineScheduler(epoch_iters=cfg.max_iter, epochs=1, warmup_iters=cfg.max_iter//10, max_lr=1e-3, min_lr=1e-6)

    for i in range(cfg.max_iter):
        aif_cp = interp_aif(parker_aif, weinmann_aif, beta_iter)
        ct_iter = evaluate_curve(ktrans_iter, kep_iter, t0_init, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=aif_cp)
        loss_iter = torch.nn.functional.l1_loss(ct_iter, ct, reduction='none').sum(dim=-1)
        loss = loss_iter.mean()

        lr = scheduler.step()
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * 1e2

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # beta_iter.data.clamp_(0, 1)
        logger.info(f"{patient_id}: [{i+1:03d}/{cfg.max_iter:03d}] lr={lr:.2e} loss={loss.item():.4f}")
    ct_iter = ct_iter.detach().cpu()
    ktrans_iter = ktrans_iter.detach().cpu()
    kep_iter = kep_iter.detach().cpu()
    loss_iter = loss_iter.detach().cpu()
    beta_iter = beta_iter.detach().cpu()
    return dict(
        patient_id=patient_id,
        ktrans_init=ktrans_init,
        kep_init=kep_init,
        t0_init=t0_init,
        ct_iter=ct_iter,
        ktrans_iter=ktrans_iter,
        kep_iter=kep_iter,
        loss_iter=loss_iter,
        beta_iter=beta_iter,
        t2w=t2w,
        cad_ktrans=cad_ktrans)


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg['checkpoint'] = args.checkpoint
    cfg['max_iter'] = args.max_iter
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'infeerence_{timestamp}.log')
    logger = get_logger(name="DCE (inference)", log_file=log_file)
    logger.info(cfg)

    np.random.seed(0)
    mmcv.runner.utils.set_random_seed(0)
    cfg['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for dce_dir in find_image_dirs.find_dce_folders('data/test-data'):
        # data = loadmat('../tmp/parker_aif/10042_1_003Tnq2B-20180212.mat')
        results = process_patient(cfg, dce_dir, save_dir='results/dispersion-1')
        if results is not None:
            patient_id = results['patient_id']
            save2dicom(results['ktrans_iter'], save_dir=f'{cfg.work_dir}/results-1221-2022/{patient_id}', example_dicom=results['t2w'], description='Ktrans-beta')
            save2dicom(results['beta_iter'], save_dir=f'{cfg.work_dir}/results-1221-2022/{patient_id}', example_dicom=results['t2w'], description='beta')

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
