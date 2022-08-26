from genericpath import isfile
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from einops import rearrange

from mmcv.cnn import MODELS
from mmcv.utils import Config, DictAction, get_logger

from scipy.io import loadmat

import os, sys, argparse, time, math, shutil, warnings
import os.path as osp

from tqdm import tqdm
import vlkit.plt as vlplt

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import load_dce
from utils import inference, str2int, save_slices_to_dicom
from pharmacokinetic import calculate_reconstruction_loss
from aif import parker_aif, biexp_aif, get_aif


def save2dicom(array, save_dir, example_dicom, name, **kwargs):
    SeriesNumber = int(str(str2int(name))[-12:])
    SeriesInstanceUID = str(SeriesNumber)
    save_dir = osp.join(save_dir, name)
    save_slices_to_dicom(
        array,
        dicom_dir=save_dir,
        example_dicom=example_dicom,
        SeriesNumber=SeriesNumber,
        SeriesInstanceUID=SeriesInstanceUID,
        SeriesDescription=name,
        **kwargs)



if __name__ == '__main__':
    work_dir = 'work_dirs/patient-results/'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dce_data = load_dce.load_dce_data('../dicom/10042_1_1Wnr0444/20161209/iCAD-MCC_33000', device=device)
    max_base = dce_data['max_base'].item()
    acquisition_time = dce_data['acquisition_time']

    cfg_parker = Config.fromfile('work_dirs/losses/loss_param-loss_ct/config.py')
    cfg_weinmann = Config.fromfile('work_dirs/AIF/weinmann-t0_0.1-0.6/weinmann_aif.py')
    cfg_fh = Config.fromfile('work_dirs/AIF/fh/config.py')

    # prepare models
    model_parker = MODELS.build(cfg_parker.model)
    model_parker.load_state_dict(torch.load(osp.join(cfg_parker.work_dir, 'model-iter50000.pth')))
    #
    model_weinmann = MODELS.build(cfg_weinmann.model)
    model_weinmann.load_state_dict(torch.load(osp.join(cfg_weinmann.work_dir, 'model-iter50000.pth')))
    #
    model_fh = MODELS.build(cfg_fh.model)
    model_fh.load_state_dict(torch.load(osp.join(cfg_fh.work_dir, 'model-iter50000.pth')))

    # prepare data
    dce_folders = load_dce.find_dce_folders('../dicom-debug')

    example_dicom = '/data1/IDX_Current/dicom/10042_1_004D6Sy8/20160616/iCAD-Ktrans-FA-0-E_30009'

    for folder in dce_folders:
        dce_data = load_dce.load_dce_data(folder)
        max_base = dce_data['max_base'].item()
        acquisition_time = dce_data['acquisition_time']
        ct = dce_data['ct'].to(device)

        patient_dir = osp.join(work_dir, dce_data['patient'])
        os.makedirs(patient_dir, exist_ok=True)

        t2dir = list(filter(lambda x: 't2' in x and 'tse' in x and 'tra' in x,  os.listdir(osp.join(folder, '../'))))
        if len(t2dir) > 0:
            t2dir = osp.abspath(osp.join(folder, '../', t2dir[0]))
        else:
            t2dir = None

        icad_ktrans_dir = list(filter(lambda x: 'ktrans' in x.lower(),  os.listdir(osp.join(folder, '../'))))
        if len(icad_ktrans_dir) >= 1:
            if any(filter(lambda x: 'CLR' not in x, icad_ktrans_dir)):
                icad_ktrans_dir = list(filter(lambda x: 'CLR' not in x, icad_ktrans_dir))
        else:
            icad_ktrans_dir = None
        icad_ktrans_dir = osp.abspath(osp.join(folder, '../', icad_ktrans_dir[0]))

        try:
            os.symlink(t2dir, osp.join(patient_dir, t2dir.split(osp.sep)[-1]))
        except:
            warnings.warn('Symlink from %s to %s failed' % (t2dir, osp.join(patient_dir, t2dir.split(osp.sep)[-1])))
        
        try:
            os.symlink(icad_ktrans_dir, osp.join(patient_dir, icad_ktrans_dir.split(osp.sep)[-1]))
        except:
            warnings.warn('Symlink from %s to %s failed' % (icad_ktrans_dir, osp.join(patient_dir, icad_ktrans_dir.split(osp.sep)[-1])))

        logger = open(osp.join(patient_dir, 'log.txt'), 'w')
        logger.write('max base: %f' % max_base)
        logger.close()

        plt.plot(dce_data['center_curve'].cpu().numpy(), color='g')
        plt.plot(dce_data['center_grad'].cpu().numpy(), color='r')
        plt.axvline(x=max_base, color='b')
        plt.legend(['curve', 'grad', 'max_base'])
        plt.title(dce_data['patient'])
        plt.grid(True)
        plt.savefig(osp.join(patient_dir, 'center-curve.pdf'))
        plt.close('all')

        # for aif in ['parker', 'weinmann', 'fh']:
        # for aif in ['weinmann']:
        for aif in ['weinmann', 'parker']:
            aif_cp, aif_t = get_aif(aif, acquisition_time, max_base)
            model = eval('model_' + aif).to(device)
            torch.cuda.empty_cache()

            ktrans, kep, t0 = inference(model, ct)
            loss = calculate_reconstruction_loss(ktrans, kep, t0, ct, acquisition_time, aif_t, aif_cp, relative=True)

            ktrans = ktrans.cpu().numpy()
            kep = kep.cpu().numpy()
            t0 = t0.cpu().numpy()
            loss = loss.cpu().numpy()

            shared_kwargs = dict(PatientID=dce_data['patient'].upper(), PatientName=dce_data['patient'].upper(), example_dicom=example_dicom)
            shared_kwargs.update(dce_data['metadata'])

            # matlab results
            matlab_results = '../tmp/%s_aif/' % aif
            matlab_results = osp.join(matlab_results, '-'.join(folder.split(osp.sep)[-3:-1]) + '.mat')
            if osp.isfile(matlab_results):
                matlab_results = loadmat(matlab_results)
            else:
                matlab_results = None

            save2dicom(ktrans, patient_dir, name='transformer-%s-aif-ktrans' % aif, **shared_kwargs)
            save2dicom(kep, patient_dir, name='transformer-%s-aif-kep' % aif, **shared_kwargs)
            save2dicom(t0, patient_dir, name='transformer-%s-aif-t0' % aif, **shared_kwargs)
            save2dicom(loss, patient_dir, name='transformer-%s-aif-loss' % aif, **shared_kwargs)

            if matlab_results is not None:
                matlab_loss = calculate_reconstruction_loss(
                    torch.from_numpy(matlab_results['ktrans']).cuda(),
                    torch.from_numpy(matlab_results['kep']).cuda(),
                    torch.from_numpy(matlab_results['t0']).cuda(),
                    ct, acquisition_time, aif_t, aif_cp, relative=True)
                save2dicom(matlab_results['ktrans'], save_dir=patient_dir, name='matlab-%s-aif-ktrans' % aif, **shared_kwargs)
                save2dicom(matlab_results['kep'], save_dir=patient_dir, name='matlab-%s-aif-kep' % aif, **shared_kwargs)
                save2dicom(matlab_results['t0'], save_dir=patient_dir, name='matlab-%s-aif-t0' % aif, **shared_kwargs)
                save2dicom(matlab_loss.cpu(), save_dir=patient_dir, name='matlab-%s-aif-loss' % aif, **shared_kwargs)