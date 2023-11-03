import matplotlib
matplotlib.use('agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import vlkit.plt as vlplt

from prettytable import PrettyTable
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
from vlkit.medical import read_dicoms
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
    parser = argparse.ArgumentParser(description='DCE inference')
    parser.add_argument('data', help='data path')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint weights')
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--max-lr', type=float, default=1e-2)
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


def save_resuls_to_dicoms(results, method, aif, saveto, example_dcm):
    save2dicom(results['ktrans'], save_dir=f'{saveto}/{patient_id}', example_dicom=example_dcm, description=f'Ktrans-{aif}-AIF-{method}')
    # save2dicom(results['kep'], save_dir=f'{saveto}/{patient_id}', example_dicom=example_dcm, description=f'kep-{aif}-AIF-{method}')
    # save2dicom(results['error'], save_dir=f'{saveto}/{patient_id}', example_dicom=example_dcm, description=f'error-{aif}-AIF-{method}')
    if 'beta' in results and results['beta'] is not None:
        save2dicom(results['beta'], save_dir=f'{saveto}/{patient_id}', example_dicom=example_dcm, description=f'beta-{method}')
        if aif == 'mixed':
            ktrans_x_beta = results['ktrans'] * results['beta']
            save2dicom(ktrans_x_beta, save_dir=f'{saveto}/{patient_id}', example_dicom=example_dcm, description=f'Ktrans-x-beta-{method}')
            #
            ktrans_x_beta = results['ktrans'] * results['beta'].exp()
            save2dicom(ktrans_x_beta, save_dir=f'{saveto}/{patient_id}', example_dicom=example_dcm, description=f'Ktrans-x-betaexp-{method}')


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


def process_patient(cfg, dce_data, init='random', refine=True, aif=None, max_iter=100, max_lr=1e-2, min_lr=1e-5):
    if aif == None:
        aif = cfg.aif

    assert init in ['nn', 'random']
    if not isinstance(dce_data, dict):
        raise RuntimeError(type(dce_data))

    ct = dce_data['ct'].to(cfg.device)
    h, w, slices, frames = ct.shape
    mask = ct.max(dim=-1).values >= ct.max(dim=-1).values.max() / 50
    N = mask.sum()
    data = ct[mask]

    if init == 'nn':
        model = MODELS.build(cfg.model).to(cfg.device)
        model.load_state_dict(torch.load(cfg.checkpoint))

        torch.cuda.empty_cache()
        tic = time.time()
        output = inference(model, data)
        toc = time.time()
        logger.info(f'NN inference ({mask.sum().item()} voxels) takes {toc-tic:.3f} seconds')
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

        scheduler = MultiStepScheduler(gammas=0.1, milestones=[int(max_iter*3/4), int(max_iter*7/8)], base_lr=max_lr)

        tic = time.time()
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
                logger.info(f"{aif} AIF [{i+1:03d}/{max_iter:03d}] lr={lr:.2e} loss={loss.item():.5f}")
        logger.info(f"Refine ({mask.sum().item()} voxels) takes {time.time()-tic:.3f} second.")

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
        ktrans=ktrans_map,
        kep=kep_map,
        t0=t0,
        ct=ct,
        loss=loss.mean().item(),
        error=error_map,
        # t2w=t2w
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

        table = PrettyTable()
        study_dir = osp.abspath(osp.join(dce_dir, '../'))
        t2_dir = find_image_dirs.find_t2_folder(study_dir)
        dynacad_dirs = find_image_dirs.find_dynacad(study_dir, include_clr=True)
        osirixsr_dir = find_image_dirs.find_osirixsr(study_dir)

        if osp.isdir(f'{args.save_path}/{patient_id}'):
            logger.info(f'{args.save_path}/{patient_id} exists, pass.')
            continue

        # try to load dce-mri data
        try:
            dce_data = load_dce.load_dce_data(dce_dir, device=cfg.device)
        except Exception as e:
            logger.warn(f"{dce_dir}: cannot load DCE data from {dce_dir}: {e}")
            logger.error(e)
            continue

        if histopathology_dir is None:
            logger.warn(f"{patient_id}: cannot find histopathology")
        if dynacad_dirs is None:
            logger.warn(f"{patient_id}: cannot find dynacad_dirs")

        table_data = {}
        for aif in ['mixed', "parker", "weinmann"]:
            logger.info(f"Process patient {patient_id}/{exp_date} with {aif} AIF")
            example_dcm = read_dicoms(dce_dir)[:dce_data['ct'].shape[-2]]
            # get our results
            logger.info(f'{patient_id}: get our results')
            results = process_patient(cfg, dce_data, aif=aif, max_iter=200, max_lr=args.max_lr)
            table_data[f'ours-{aif}'] = results['loss']

            if results is None:
                logger.warn(f'Patient {patient_id} result is .')
                continue
            save_resuls_to_dicoms(results=results, aif=aif, method='ours', saveto=args.save_path, example_dcm=example_dcm)

            # get NLLS results
            logger.info(f'{patient_id}: get NLLS results')
            results = process_patient(cfg, dce_data, aif=aif, init='random', max_iter=80, max_lr=args.max_lr)
            save_resuls_to_dicoms(results=results, aif=aif, method='NLLS', saveto=args.save_path, example_dcm=example_dcm)
            table_data[f'NLLS-{aif}'] = results['loss']
        for k, v in table_data.items():
            table.add_column(k, [v])
        logger.info(table)

        if histopathology_dir is not None:
            dst = osp.join(f'{args.save_path}/{patient_id}', 'histopathology')
            shutil.copytree(histopathology_dir, dst, dirs_exist_ok=True)

        if dynacad_dirs is not None:
            for d in dynacad_dirs:
                dst = osp.join(f'{args.save_path}/{patient_id}', d.split(os.sep)[-1])
                shutil.copytree(d, dst)