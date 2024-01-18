import torch, math
import numpy as np
import pydicom, os, warnings, time
import os.path as osp
from pydicom.dataset import Dataset
from vlkit.image import norm01
from einops import rearrange
from tqdm import tqdm
from collections import OrderedDict
from tofts import tofts
from pydicom.uid import ExplicitVRLittleEndian

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


def save_slices_to_dicom(data, save_dir, example_dicom=None, **kwargs):
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

    if isinstance(example_dicom, str) and osp.isdir(example_dicom):
        example_dicom = sorted([osp.join(example_dicom, i) for i in os.listdir(example_dicom) if i.endswith('dcm')])
        example_dicom = [pydicom.dcmread(d) for d in example_dicom]
    elif isinstance(example_dicom, list):
        pass
    else:
        raise ValueError

    for i in range(slices):
        save_fn = osp.join(save_dir, 'slice-%.3d.dcm' % (i+1))
        img = np.squeeze(data_uint16[:, :, i])
        write_dicom(
            img,
            save_fn,
            ds=example_dicom[i],
            **kwargs
        )


def write_dicom(array, filename, ds, **kwargs):
    assert isinstance(array, np.ndarray)
    assert array.ndim == 2
    os.makedirs(osp.dirname(filename), exist_ok=True)

    ds.Rows = array.shape[0]
    ds.Columns = array.shape[1]
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    if kwargs is not None:
        for k, v in kwargs.items():
            setattr(ds, k, v)

    ds.PixelData = array.tobytes()

    ds.save_as(filename, write_like_original=False)


def save2dicom(array, save_dir, example_dicom, description, **kwargs):
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


def ct_loss(pred, target, reg_neg=10, loss_func=torch.nn.functional.l1_loss):
    assert reg_neg >= 0
    loss = loss_func(pred, target)
    reg = -reg_neg * ((pred < 0) * pred).mean()
    return loss + reg


def mix_l1_mse_loss(x, y):
    return torch.nn.functional.l1_loss(x, y) / 2 + torch.nn.functional.mse_loss(x, y) / 2


def spatial_loss(data, uncertainty, r=3, alpha=-0.1):
    assert data.ndim == 3
    loss = torch.zeros_like(data)
    h, w, c = data.shape
    sq = (torch.arange(-r, r+1).view(-1, 1) ** 2 + torch.arange(-r, r+1).view(1, -1) ** 2).sqrt().unsqueeze(dim=-1)
    sq = sq.neg().multiply(0.5).exp().to(data)
    for y in range(r, h-r):
        for x in range(r, w-r):
            data1 = data[y-r:y+r+1, x-r:x+r+1, :]
            u = uncertainty[y-r:y+r+1, x-r:x+r+1, :]
            reference = data[y, x].view(1, 1, -1).expand_as(data1)
            l = torch.nn.functional.mse_loss(data1, reference, reduction='none')
            l = (l * sq * u).sum(dim=(0, 1))
            loss[y, x, :] = l
    return loss


# model related
def inference(model, data, batchsize=256, convert2cpu=False):
    assert data.ndim == 5 or data.ndim == 4 or data.ndim == 2, data.shape
    if data.ndim == 4:
        # h w slices frames
        h, w, slices, frames = data.shape
        output_shape = [h, w, slices]
        data = data.reshape(h*w*slices, frames, 1)
    elif data.ndim == 5:
        # h w slices frames 1
        assert data.size(-1) == 5
        h, w, slices, frames, _ = data.shape
        output_shape = [h, w, slices]
        data = data.reshape(h*w*slices, frames, 1)
    elif data.ndim == 2:
        n, frames = data.shape
        output_shape = [n]
        data = data.reshape(-1, frames, 1)
    chunks = math.ceil(data.size(0) / batchsize)
    data = data.chunk(dim=0, chunks=chunks)
    model.eval()

    output = OrderedDict()
    for k in model.output_keys:
        output[k] = []

    

    print('Start inference...')
    tic = time.time()
    with torch.no_grad():
        for i, data1 in enumerate(tqdm(data)):
            o = model(data1)
            for k in model.output_keys:
                output[k].append(o[k])
    toc = time.time()
    for k in model.output_keys:
        output[k] = torch.cat(output[k], dim=0)
    for k in model.output_keys:
        output[k] = output[k].view(*output_shape)
    print('Done, %.3fs elapsed.' % (toc-tic))
    return output