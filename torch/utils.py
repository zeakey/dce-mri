import torch, math
import numpy as np
import pydicom, os, warnings
import os.path as osp
from pydicom.dataset import Dataset
from vlkit.image import norm01
from einops import rearrange


def load_dce(folder, shift_max_base=False):
    data = read_dce_folder(folder)
    dce_data = torch.tensor(data['data'].astype(np.float32))
    acquisition_time = torch.tensor(data['acquisition_time'])
    repetition_time = torch.tensor(data['repetition_time'])
    flip_angle = torch.tensor(data['flip_angle'])

    max_base = find_max_base(dce_data)

    if shift_max_base:
        # important! shift max_base for uniform ct
        dce_data = dce_data[:, :, :, max_base:]
        dce_data = torch.cat((
                dce_data,
                dce_data[:, :, :, -1:].repeat(1, 1, 1, max_base)
            ), dim=-1)

        acquisition_time = acquisition_time[max_base:]
        interval = (acquisition_time[-1] - acquisition_time[0]) / (acquisition_time.numel() - 1)

        acquisition_time = torch.cat((acquisition_time, acquisition_time[-1] + torch.arange(1, max_base+1) * interval), dim=0)
        max_base = 0

    # second to minite
    acquisition_time = acquisition_time - acquisition_time[0]
    acquisition_time = acquisition_time / 60

    ct = gd_concentration(dce_data, max_base=0)

    return dict(
        ct=ct,
        flip_angle=flip_angle,
        repetition_time=repetition_time,
        acquisition_time=acquisition_time,
        max_base=max_base,
    )


def read_dce_folder(folder):
    dicoms = sorted([i for i in os.listdir(folder) if i.endswith('dcm') or i.endswith('dicom')])
    data = []
    flip_angle = []
    repetition_time = []
    acquisition_time = []
    for d in dicoms:
        ds = pydicom.dcmread(osp.join(folder, d))
        acquisition_time.append(60 * (float(ds.AcquisitionTime[0:2]) * 60 + float(ds.AcquisitionTime[2:4])) + float(ds.AcquisitionTime[4:]))
        data.append(ds.pixel_array)
        flip_angle.append(float(ds.FlipAngle))
        repetition_time.append(float(ds.RepetitionTime))
    acquisition_time = np.array(acquisition_time).reshape(75, 20)[:, 0]
    data = np.stack(data, axis=-1)
    h, w, _ = data.shape
    data = rearrange(data, 'h w (frames slices) -> h w slices frames', h=h, w=w, slices=20, frames=75)
    flip_angle = np.array(flip_angle)
    repetition_time = np.array(repetition_time)
    return dict(data=data, flip_angle=flip_angle, repetition_time=repetition_time, acquisition_time=acquisition_time)


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
    return bnum


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


def save_slices_to_dicom(data, dicom_dir, SeriesDescription, **kwargs):
    """
    data: [h w slices]
    """
    if data.min() < 0:
        data[data < 0] = 0
        warnings.warn('Negative values are clipped to 0.')
    
    kwargs['SeriesDescription'] = SeriesDescription

    # identify file type from SeriesDescription
    if 'ktrans' in SeriesDescription.lower():
        ftype = 'ktrans'
    elif 'kep' in SeriesDescription.lower():
        ftype = 'kep'
    elif 't0' in SeriesDescription.lower():
        ftype = 't0'
    else:
        warnings.warn('cannot identify ftype from SeriesDescription=%s' % SeriesDescription)

    data = data.astype(np.float64)
    data_uint16 = (data * 1000).astype(np.uint16)
    slices = data_uint16.shape[2]
    example_dicom_dir = osp.join(osp.dirname(__file__), '../example-dicom', ftype)
    example_dicoms = sorted([osp.join(example_dicom_dir, i) for i in os.listdir(example_dicom_dir) if i.endswith('dcm')])
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