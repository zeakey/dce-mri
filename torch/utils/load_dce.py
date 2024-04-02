import os, torch, pydicom
from collections import OrderedDict
import numpy as np
import os.path as osp
from einops import rearrange
from tqdm import tqdm

from pharmacokinetic import find_max_base, gd_concentration


def load_dce_data(folder, shift_max_base=False, device=torch.device('cpu')):
    dicoms = [i for i in os.listdir(folder) if osp.isfile(osp.join(folder, i))]
    if folder.endswith(os.sep):
        folder = folder[:-1]
    # metadata
    metadata = dict()
    ds = pydicom.dcmread(osp.join(folder, os.listdir(folder)[0]))
    metadata['AccessionNumber'] = ds.AccessionNumber
    metadata['AcquisitionDate'] = ds.AcquisitionDate
    metadata['StudyDate'] = ds.StudyDate
    metadata['SeriesDate'] = ds.SeriesDate
    metadata['ContentDate'] = ds.ContentDate
    metadata['SOPClassUID'] = ds.SOPClassUID
    metadata['SOPInstanceUID'] = ds.SOPInstanceUID
    if hasattr(ds, "StudyID"):
        metadata['StudyID'] = ds.StudyID
    else:
        metadata['StudyID'] = "Placeholder-studyID"
    metadata['StudyInstanceUID'] = ds.StudyInstanceUID
    data = read_dce_dicoms(folder)
    dce_data = torch.tensor(data['data'].astype(np.float32)).to(device)
    acquisition_time = torch.tensor(data['acquisition_time']).to(dce_data)
    repetition_time = torch.tensor(data['repetition_time']).to(dce_data)
    flip_angle = torch.tensor(data['flip_angle']).to(dce_data)

    max_base, center_curve, center_grad = find_max_base(dce_data)

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

    return OrderedDict(
        ct=ct,
        flip_angle=flip_angle,
        repetition_time=repetition_time,
        acquisition_time=acquisition_time,
        max_base=max_base,
        center_curve=center_curve,
        center_grad=center_grad,
        metadata=metadata)


def parse_acquisition_time(acquisition_time):
    return 60 * (float(acquisition_time[0:2]) * 60 + float(acquisition_time[2:4])) + float(acquisition_time[4:])


def read_dce_dicoms(folder):
    dicoms = sorted([osp.join(folder, i) for i in os.listdir(folder) if i.endswith('dcm') or i.endswith('dicom')])
    flip_angle = []
    repetition_time = []
    acquisition_time = []
    data = []
    for d in tqdm(dicoms):
        ds = pydicom.dcmread(d)
        data.append(ds.pixel_array)
        acquisition_time.append(parse_acquisition_time(ds.AcquisitionTime))
        flip_angle.append(float(ds.FlipAngle))
        repetition_time.append(float(ds.RepetitionTime))
    num_frames = len(np.unique(acquisition_time))
    num_slices = len(acquisition_time) // num_frames
    acquisition_time = np.array(acquisition_time).reshape(num_frames, num_slices)[:, 0]
    data = np.stack(data, axis=-1)
    h, w, _ = data.shape
    data = rearrange(data, 'h w (frames slices) -> h w slices frames', h=h, w=w, slices=num_slices, frames=num_frames)
    flip_angle = np.array(flip_angle)
    repetition_time = np.array(repetition_time)
    return dict(
        data=data,
        flip_angle=flip_angle,
        repetition_time=repetition_time,
        acquisition_time=acquisition_time,
        frames=num_frames,
        slices=num_slices
    )
