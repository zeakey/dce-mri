from genericpath import isdir
import torch, math
import numpy as np
import pydicom, os, warnings, time
import os.path as osp
from pydicom.dataset import Dataset
from vlkit.image import norm01
from einops import rearrange
from tqdm import tqdm

from aif import parker_aif, biexp_aif


def str2int(s):
    return int.from_bytes(s.encode(), 'little')

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


def  normalize_dce(folder):
    data = read_dce_folder(folder)
    max_base = find_max_base(data['data'])
    data['data'] = data['data'][:, :, :, max_base:]


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


def read_dicom_data(path):
    if osp.isdir(path):
        dicoms = [i for i in os.listdir(path) if i.endswith('dcm') or i.endswith('dicom')]
        dicoms = sorted(dicoms)
    elif osp.isfile(path):
        dicoms = [path]
    else:
        raise ValueError

    data = []
    for d in dicoms:
        ds = pydicom.dcmread(osp.join(path, d))
        ds = ds.pixel_array
        data.append(ds)
    return data


def save_slices_to_dicom(data, dicom_dir, example_dicom=None, **kwargs):
    """
    data: [h w slices]
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if data.min() < 0:
        data[data < 0] = 0
        warnings.warn('Negative values are clipped to 0.')

    data = data.astype(np.float64)
    data_uint16 = (data * 1000).astype(np.uint16)
    slices = data_uint16.shape[2]
    if example_dicom is None:
        example_dicoms = '/data1/IDX_Current/dicom/10042_1_004D6Sy8/20160616/iCAD-MCC-Ktrans-FA-0-E_33009/'

    if osp.isdir(example_dicom):
        example_dicoms = sorted([osp.join(example_dicom, i) for i in os.listdir(example_dicom) if i.endswith('dcm')])
    else:
        raise ValueError

    for i in range(slices):
        dicom = example_dicoms[i]
        dicom = pydicom.dcmread(dicom)
        save_fn = osp.join(dicom_dir, 'slice-%.3d.dcm' % (i+1))
        img = np.squeeze(data_uint16[:, :, i])
        thickness = 3.6
        write_dicom(
            img,
            save_fn,
            ds=dicom,
            **kwargs
        )


def write_dicom(array, filename, ds, **kwargs):
    assert isinstance(array, np.ndarray)
    assert array.ndim == 2
    os.makedirs(osp.dirname(filename), exist_ok=True)

    ds.Rows = array.shape[0]
    ds.Columns = array.shape[1]

    if kwargs is not None:
        for k, v in kwargs.items():
            setattr(ds, k, v)

    ds.PixelData = array.tobytes()

    ds.save_as(filename, write_like_original=False)


# model related
def inference(model, data):
    assert data.ndim == 4
    model.eval()

    h, w, s, f = data.shape
    data = rearrange(data, 'h w s f -> h (w s) f')
    ktrans = torch.zeros(h, w*s).to(data)
    kep = torch.zeros(h, w*s).to(data)
    t0 = torch.zeros(h, w*s).to(data)

    print('Start inference...')
    tic = time.time()
    with torch.no_grad():
        for i, d in enumerate(tqdm(data)):
            ktrans_, kep_, t0_ = model(d.unsqueeze(dim=-1))
            ktrans[i, ] = ktrans_.squeeze(dim=1)
            kep[i, ] = kep_.squeeze(dim=1)
            t0[i, ] = t0_.squeeze(dim=1)
    toc = time.time()
    print('Done, %.3fs elapsed.' % (toc-tic))
    ktrans = rearrange(ktrans, 'h (w s) -> h w s', w=w)
    kep = rearrange(kep, 'h (w s) -> h w s', w=w)
    t0 = rearrange(t0, 'h (w s) -> h w s', w=w)
    return ktrans, kep, t0


def get_aif(aif: str, acquisition_time: torch.Tensor, max_base: int, hct: float=0.42):
    assert aif in ['parker', 'weinmann', 'fh']
    aif_t = torch.arange(0, acquisition_time[-1], 1/60).to(acquisition_time)

    if aif == 'parker':
        aif_cp = parker_aif(
            a1=0.809,
            a2=0.330,
            t1=0.17046,
            t2=0.365,
            sigma1=0.0563,
            sigma2=0.132,
            alpha=1.050,
            beta=0.1685,
            s=38.078,
            tau=0.483,
            t=aif_t - (acquisition_time[max_base] / (1 / 60)).ceil() * (1 / 60)
        ) / (1 - hct)
    elif aif == 'weinmann':
        aif_cp = biexp_aif(
            3.99,
            4.78,
            0.144,
            0.011,
            aif_t - (acquisition_time[max_base] / (1 / 60)).ceil() * (1 / 60)
        ) / (1 - hct)
    elif aif == 'fh':
        aif_cp = biexp_aif(
            24,
            6.2,
            3.0,
            0.016,
            aif_t
        ) / (1 - hct)
    return aif_cp, aif_t