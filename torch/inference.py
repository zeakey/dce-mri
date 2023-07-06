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
from mmcv.utils import Config, DictAction
from mmcv.runner import set_random_seed

from tqdm import tqdm
from einops import rearrange

import mmcv
from vlkit.image import normalize
from vlkit.dicom.dicomio import read_dicoms
from vlkit.lrscheduler import CosineScheduler, MultiStepScheduler
from vlkit.utils import   get_logger

from scipy.io import loadmat, savemat
import pydicom

import time, sys, os, shutil, argparse
import os.path as osp
sys.path.insert(0, '/data1/Code/vlkit/vlkit/medical')
sys.path.insert(0, '..')

from utils import inference, save_slices_to_dicom, str2int
from aif import get_aif, interp_aif
from pharmacokinetic import process_patient, evaluate_curve
from utils import load_dce, find_image_dirs
from pharmacokinetic import calculate_reconstruction_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('data', help='data path')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint weights')
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--save-path', type=str, default='/tmp/dce-mri')
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


def process_patient(cfg, dce_dir, init='nn', refine=True, aif=None, max_iter=100, max_lr=1e-3, min_lr=1e-6):
    if aif == None:
        aif = cfg.aif

    assert init in ['nn', 'random']

    patient_id = dce_dir.split(os.sep)[-3]
    study_dir = osp.join(dce_dir, '../')
    cad_ktrans_dir = find_image_dirs.find_ktrans_folder(study_dir)
    t2_dir = find_image_dirs.find_t2_folder(study_dir)

    if cad_ktrans_dir and t2_dir and osirixsr_dir:
        t2w = read_dicoms(t2_dir)
        cad_ktrans = read_dicoms(cad_ktrans_dir)
    else:
        logger.warn(f"{patient_id}: cannot find t2w or cad_ktrans, pass")
        return None
    try:
        dce_data = load_dce.load_dce_data(dce_dir, device=cfg.device)
    except Exception as e:
        logger.warn(f"Cannot load DCE data from {dce_dir}: {e}")
        return None

    # get the mask determining which voxels will be processed
    # time series whose maximal value is less then 1/100 times the global maximum will be ignored
    ct = dce_data['ct'].to(cfg.device)
    h, w, slices, frames = ct.shape
    mask = ct.max(dim=-1).values >= ct.max(dim=-1).values.max() / 100
    N = mask.sum()
    data = ct[mask]

    if init == 'nn':
        model = MODELS.build(cfg.model).to(cfg.device)
        model.load_state_dict(torch.load(cfg.checkpoint))

        torch.cuda.empty_cache()
        tic = time.time()
        output = inference(model, data)
        toc = time.time()
        logger.info('%.2f seconds' % (toc-tic))
        del model

        ktrans = output['ktrans']
        kep = output['kep']
        t0 = output['t0']
        if aif == 'mixed':
            beta = output['beta']
        ktrans = ktrans.squeeze(-1)
        kep = kep.squeeze(-1)
        t0 = t0.squeeze(-1)
    elif init == 'random':
        shape = dce_data['ct'].shape[:-1]
        ktrans = torch.rand(data.size(0)).to(cfg.device)
        kep = torch.rand(data.size(0)).to(ktrans)
        t0 = torch.rand(data.size(0)).to(ktrans)
        if aif == 'mixed':
            beta = torch.rand(data.size(0)).to(ktrans)
    else:
        raise ValueError(init)

    if aif == 'mixed':
        weinmann_aif, aif_t = get_aif(aif='weinmann', max_base=6, acquisition_time=dce_data['acquisition_time'], device=cfg.device)
        parker_aif, _ = get_aif(aif='parker', max_base=6, acquisition_time=dce_data['acquisition_time'], device=cfg.device)
        beta = beta.squeeze(-1)
        aif_cp = interp_aif(parker_aif, weinmann_aif, beta=beta)
    else:
        aif_cp, aif_t = get_aif(aif=aif, max_base=6, acquisition_time=dce_data['acquisition_time'], device=cfg.device)

    if refine:
        logger.info("Start iterative refinement.")
        torch.cuda.empty_cache()

        # -------------------------------------------------------------------------------------------- #
        ktrans = ktrans.to(device=cfg.device).requires_grad_()
        kep = kep.to(device=cfg.device).requires_grad_()
        t0 = t0.to(device=cfg.device).requires_grad_()

        if aif == 'mixed':
            beta = beta.requires_grad_()
            optimizer = torch.optim.RMSprop(params=[{'params': [ktrans, kep, t0], 'lr_factor': 1}, {'params': [beta], 'lr_factor': 1e2}])
        else:
            optimizer = torch.optim.RMSprop(params=[ktrans, kep, t0], lr=1e-3)
            optimizer.param_groups[0]['lr_factor'] = 1

        aif_t = aif_t.to(ktrans)
        data = data.to(ktrans)

        # scheduler = CosineScheduler(epoch_iters=max_iter, epochs=1, warmup_iters=0, max_lr=max_lr, min_lr=min_lr)
        scheduler = MultiStepScheduler(iters=max_iter, gamma=0.1, milestones=[int(max_iter*3/4), int(max_iter*7/8)], base_lr=max_lr)

        for i in range(max_iter):
            if aif == 'mixed':
                aif_cp = interp_aif(parker_aif, weinmann_aif, beta)
            ct_hat = evaluate_curve(ktrans, kep, t0, t=dce_data['acquisition_time'], aif_t=aif_t, aif_cp=aif_cp)
            error = torch.nn.functional.l1_loss(ct_hat, data, reduction='none').sum(dim=-1)
            loss = error.mean()

            lr = scheduler.step()
            for pg in optimizer.param_groups:
                pg['lr'] = lr * pg['lr_factor']

            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
            if aif == 'mixed':
                beta.data.clamp_(0, 1)
            if (i+1) % 10 == 0:
                logger.info(f"{patient_id}: [{i+1:03d}/{max_iter:03d}] lr={lr:.2e} loss={loss.item():.5f}")

    ktrans_map = torch.zeros(h, w, slices).to(ktrans)
    kep_map = torch.zeros(h, w, slices).to(ktrans)
    t0_map = torch.zeros(h, w, slices).to(ktrans)
    error_map = torch.zeros(h, w, slices).to(ktrans)

    ktrans_map[mask] = ktrans
    kep_map[mask] = kep
    t0_map[mask] = t0
    error_map[mask] = error

    if aif == 'mixed':
        beta_map = torch.zeros(h, w, slices).to(ktrans)
        beta_map[mask] = beta

    results = dict(
        patient_id=patient_id,
        ktrans=ktrans_map,
        kep=kep_map,
        t0=t0,
        ct=ct,
        error=error_map,
        t2w=t2w
    )
    if aif == 'mixed':
        results['beta'] = beta_map

    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            results[k] = v.detach().cpu()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg['checkpoint'] = args.checkpoint
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.save_path, f'infeerence_{timestamp}.log')
    logger = get_logger(name="DCE (inference)", log_file=log_file)
    logger.info(cfg)

    np.random.seed(0)
    set_random_seed(0)
    cfg['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.save_path is None:
        args.save_path = f'{cfg.work_dir}/results-{timestamp}/'

    logger.info(f'Save results to {args.save_path}')
    for idx, dce_dir in enumerate(find_image_dirs.find_dce_folders(args.data)):
        patient_id, exp_date = dce_dir.split(os.sep)[-3:-1]
        histopathology_dir = find_image_dirs.find_histopathology(patient_id, exp_date)

        study_dir = osp.abspath(osp.join(dce_dir, '../'))
        iCAD_Ktrans_dir = find_image_dirs.find_icad_ktrans(study_dir)
        osirixsr_dir = find_image_dirs.find_osirixsr(study_dir)

        if histopathology_dir is None or iCAD_Ktrans_dir is None or osirixsr_dir is None:
            if histopathology_dir:
                logger.warn(f"{patient_id}: cannot find histopathology")
            if iCAD_Ktrans_dir:
                logger.warn(f"{patient_id}: cannot find iCAD_Ktrans_dir")
            continue

        if osp.isdir(f'{args.save_path}/{patient_id}'):
            logger.info(f'{args.save_path}/{patient_id} exists, pass.')
            continue

        for aif in ['mixed', 'parker', 'weinmann']:
            logger.info(f"Process patient {patient_id}/{exp_date} with {aif} AIF")

            # get our results
            results = process_patient(cfg, dce_dir, aif=aif, max_iter=120, max_lr=5e-3)
            if results is None:
                logger.warn(f'Patient {patient_id} result is .')
                continue
            t2w = torch.from_numpy(np.concatenate([i.pixel_array[:, :, None] for i in results['t2w']], axis=-1).astype(np.float32))
            t2w = torch.nn.functional.interpolate(rearrange(t2w, "h w c -> 1 c h w"), size=(160, 160))
            t2w = rearrange(t2w, '1 c h w -> h w c').numpy()
            ct = results['ct']
            ktrans = results['ktrans']
            kep = results['kep']
            error = results['error']
            if 'beta' in results:
                beta = results['beta']
            else:
                beta = None

            example_dcm = read_dicoms(dce_dir)[:20]
            save2dicom(ktrans, save_dir=f'{args.save_path}/{patient_id}', example_dicom=example_dcm, description=f'Ktrans-{aif}-AIF-ours')
            save2dicom(kep, save_dir=f'{args.save_path}/{patient_id}', example_dicom=example_dcm, description=f'kep-{aif}-AIF-ours')
            save2dicom(error, save_dir=f'{args.save_path}/{patient_id}', example_dicom=example_dcm, description=f'error-{aif}-AIF-ours')
            if beta is not None:
                save2dicom(beta, save_dir=f'{args.save_path}/{patient_id}', example_dicom=example_dcm, description='beta-ours')
                if aif == 'mixed':
                    ktrans_x_beta = ktrans * beta.exp()
                    save2dicom(ktrans_x_beta, save_dir=f'{args.save_path}/{patient_id}', example_dicom=example_dcm, description='Ktrans-x-beta-ours')

            # get NLLS results
            results = process_patient(cfg, dce_dir, aif=aif, init='random', max_iter=100, max_lr=1e-2)
            save2dicom(ktrans, save_dir=f'{args.save_path}/{patient_id}', example_dicom=example_dcm, description=f'Ktrans-{aif}-AIF-NLLS')
            save2dicom(kep, save_dir=f'{args.save_path}/{patient_id}', example_dicom=example_dcm, description=f'kep-{aif}-AIF-NLLS')
            save2dicom(error, save_dir=f'{args.save_path}/{patient_id}', example_dicom=example_dcm, description=f'error-{aif}-AIF-NLLS')
            if beta is not None:
                save2dicom(beta, save_dir=f'{args.save_path}/{patient_id}', example_dicom=example_dcm, description='beta-NLLS')
                beta = normalize(beta, upper_bound=1)
                if aif == 'mixed':
                    ktrans_x_beta = ktrans * beta
                    save2dicom(ktrans_x_beta, save_dir=f'{args.save_path}/{patient_id}', example_dicom=example_dcm, description='Ktrans-x-beta-NLLS')

        dst = osp.join(f'{args.save_path}/{patient_id}', 'histopathology')
        if not osp.isdir(histopathology_dir):
            print('?')
        shutil.copytree(histopathology_dir, dst, dirs_exist_ok=True)

        shutil.copytree(osirixsr_dir, osp.join(f'{args.save_path}/{patient_id}', 'OSIRIX_SR'), dirs_exist_ok=True)
        for d in iCAD_Ktrans_dir:
            dst = osp.join(f'{args.save_path}/{patient_id}', d.split(os.sep)[-1])
            shutil.copytree(d, dst)
