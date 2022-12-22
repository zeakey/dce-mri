import matplotlib
matplotlib.use('agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import vlkit.plt as vlplt

from collections import OrderedDict

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
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--clip-beta', action='store_true')
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


def process_patient(cfg, dce_dir, optimize_beta=False):
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

    ct_init = evaluate_curve(ktrans_init, kep_init, t0_init, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=aif_cp).cpu()
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

    if optimize_beta:
        beta_iter = torch.ones_like(kep_iter).requires_grad_()
        optimizer = torch.optim.RMSprop(params=[{'params': [ktrans_iter, kep_iter], 'lr': 1e-1}, {'params': [beta_iter], 'lr': 1e-3}])
        optimizer.param_groups[0]['lr_factor'] = 1
        optimizer.param_groups[1]['lr_factor'] = 1e2
    else:
        optimizer = torch.optim.RMSprop(params=[ktrans_iter, kep_iter], lr=1e-3)
        optimizer.param_groups[0]['lr_factor'] = 1

    aif_t = aif_t.to(ktrans_iter)
    ct = ct.to(ktrans_iter)
    
    scheduler = CosineScheduler(epoch_iters=cfg.max_iter, epochs=1, warmup_iters=cfg.max_iter//10, max_lr=1e-3, min_lr=1e-6)

    for i in range(cfg.max_iter):
        if optimize_beta:
            aif_cp = interp_aif(parker_aif, weinmann_aif, beta_iter)
        else:
            aif_cp = parker_aif
        ct_iter = evaluate_curve(ktrans_iter, kep_iter, t0_init, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=aif_cp)
        loss_iter = torch.nn.functional.l1_loss(ct_iter, ct, reduction='none').sum(dim=-1)
        loss = loss_iter.mean()

        lr = scheduler.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr * pg['lr_factor']

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if cfg.clip_beta:
            beta_iter.data.clamp_(0, 1)
        logger.info(f"{patient_id}: [{i+1:03d}/{cfg.max_iter:03d}] lr={lr:.2e} loss={loss.item():.4f}")

    results = dict(
        patient_id=patient_id,
        ktrans_init=ktrans_init.detach().cpu(),
        kep_init=kep_init.detach().cpu(),
        t0_init=t0_init.detach().cpu(),
        ct=ct.detach().cpu(),
        ct_init=ct_init.detach().cpu(),
        ct_iter=ct_iter.detach().cpu(),
        ktrans_iter=ktrans_iter.detach().cpu(),
        kep_iter=kep_iter.detach().cpu(),
        loss_iter=loss_iter.detach().cpu(),
        t2w=t2w,
        cad_ktrans=cad_ktrans)
    if optimize_beta:
        results['aif_cp'] = aif_cp.detach().cpu()
        results['parker_aif'] = parker_aif.detach().cpu()
        results['weinmann_aif'] = weinmann_aif.detach().cpu()
        results['beta_iter'] = beta_iter.detach().cpu()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg['checkpoint'] = args.checkpoint
    cfg['max_iter'] = args.max_iter
    cfg['clip_beta'] = args.clip_beta
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'infeerence_{timestamp}.log')
    logger = get_logger(name="DCE (inference)", log_file=log_file)
    logger.info(cfg)

    np.random.seed(0)
    mmcv.runner.utils.set_random_seed(0)
    cfg['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    save_dir = f'{cfg.work_dir}/results-1221-2022/'
    for dce_dir in find_image_dirs.find_dce_folders('data/test-data'):
        results = process_patient(cfg, dce_dir)
        results_beta = process_patient(cfg, dce_dir, optimize_beta=True)
        if results is not None:
            patient_id = results['patient_id']
            save_dir1 = f'{save_dir}/{patient_id}'
            if osp.isdir(save_dir1):
                logger.info(f'{save_dir1} exists, pass.')
                continue

            t2w = torch.from_numpy(np.concatenate([i.pixel_array[:, :, None] for i in results['t2w']], axis=-1).astype(np.float32))
            t2w = torch.nn.functional.interpolate(rearrange(t2w, "h w c -> 1 c h w"), size=(160, 160))
            t2w = rearrange(t2w, '1 c h w -> h w c').numpy()
            ct = results['ct']
            ct_init = results['ct_init']
            ct_iter = results['ct_iter']
            ct_iter_beta = results_beta['ct_iter']
            beta = results_beta['beta_iter']
            parker_aif = results_beta['parker_aif']
            weinmann_aif = results_beta['weinmann_aif']
            aif = results_beta['aif_cp']
            results_beta['beta_iter'] -= results_beta['beta_iter'].min()

            save2dicom(results['ktrans_iter'], save_dir=save_dir1, example_dicom=results['t2w'], description='Ktrans')
            save2dicom(results_beta['ktrans_iter'], save_dir=save_dir1, example_dicom=results['t2w'], description='Ktrans-beta')
            save2dicom(results_beta['beta_iter'], save_dir=save_dir1, example_dicom=results['t2w'], description='beta')

            h, w = results['ktrans_iter'].shape[:2]
            n = 100
            mask = torch.zeros(h, w)
            y_t, y_b = 70, 120
            x_l, x_r = 60, 100
            mask[y_t:y_b, x_l:x_r] = 1
            mask = mask.nonzero()
            selected = mask[torch.randperm(mask.size(0))[:n]]
            ncol = 7
            fig, axes = plt.subplots(n, ncol, figsize=(ncol*4, n*4))
            vlplt.clear_ticks(axes)

            kv = OrderedDict(
                t2w=t2w,
                ktrans_init=results['ktrans_init'].numpy(),
                ktrans_iter=results['ktrans_iter'].numpy(),
                ktrans_iter_beta=results_beta['ktrans_iter'].numpy(),
                beta =results_beta['beta_iter'].numpy())

            for i in range(n):
                y, x = selected[i]
                z = np.random.choice(range(6, 15))
                axes[i, 0].plot(ct[y, x, z, :], color='black')
                axes[i, 0].plot(ct_init[y, x, z, :], color='blue')
                axes[i, 0].plot(ct_iter[y, x, z, :], color='green')
                axes[i, 0].plot(ct_iter_beta[y, x, z, :], color='pink')
                axes[i, 0].legend(['data', 'init', 'iter', f'iter ($\\beta$={beta[y, x, z]:.2f})'])

                axes[i, 1].plot(parker_aif, color='r')
                axes[i, 1].plot(weinmann_aif, color='g')
                axes[i, 1].plot(aif[y, x, z, :], color='black')
                axes[i, 1].legend(['Parker', 'Weinmann', 'Interp'])

                for j, (k, v) in enumerate(kv.items()):
                    axes[i, j+2].imshow(norm01(v[:, :, z]))
                    roi = matplotlib.patches.Rectangle((x_l, y_t), width=x_r-x_l, height=y_b-y_t, edgecolor='r', facecolor='none')
                    axes[i, j+2].scatter(x, y, marker='x', color='red')
                    axes[i, j+2].add_patch(roi)
                    axes[i, j+2].set_title(k, fontsize=24)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{patient_id}/ct.pdf')
