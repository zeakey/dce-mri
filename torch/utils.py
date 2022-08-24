import torch, math
import numpy as np
import pydicom, os, warnings, time
import os.path as osp
from pydicom.dataset import Dataset
from vlkit.image import norm01
from einops import rearrange
from tqdm import tqdm
from tofts import tofts

from aif import parker_aif, biexp_aif


def str2int(s):
    return int.from_bytes(s.encode(), 'little')


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
def inference(model, data, convert2cpu=False):
    assert data.ndim == 5 or data.ndim == 4, data.shape
    if data.ndim == 4:
        data = data.unsqueeze(dim=-1)
    model.eval()

    h, w, s, f, d = data.shape
    data = rearrange(data, 'h w s f d -> h (w s) f d')
    ktrans = torch.zeros(h, w*s).to(data)
    kep = torch.zeros(h, w*s).to(data)
    t0 = torch.zeros(h, w*s).to(data)

    print('Start inference...')
    tic = time.time()
    with torch.no_grad():
        for i, data1 in enumerate(tqdm(data)):
            ktrans_, kep_, t0_ = model(data1)
            ktrans[i, ] = ktrans_.squeeze(dim=1)
            kep[i, ] = kep_.squeeze(dim=1)
            t0[i, ] = t0_.squeeze(dim=1)
    toc = time.time()
    print('Done, %.3fs elapsed.' % (toc-tic))
    ktrans = rearrange(ktrans, 'h (w s) -> h w s', w=w)
    kep = rearrange(kep, 'h (w s) -> h w s', w=w)
    t0 = rearrange(t0, 'h (w s) -> h w s', w=w)
    if convert2cpu:
        ktrans = ktrans.cpu()
        kep = kep.cpu()
        t0 = t0.cpu()
    return ktrans, kep, t0


