import os, torch, pydicom, warnings
from vlkit.medical.dicomio import read_dicom_array
from pydoc import isdata
from collections import OrderedDict
import numpy as np
import os.path as osp
from einops import rearrange
from tqdm import tqdm

from pharmacokinetic import find_max_base, gd_concentration


def load_dce_data(folder, shift_max_base=False, device=torch.device('cpu')):
    assert len([i for i in os.listdir(folder) if osp.isfile(osp.join(folder, i))]) == 20*75, osp.abspath(folder)
    if folder.endswith(os.sep):
        folder = folder[:-1]
    patient = folder.split(os.sep)[-3]
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
    metadata['StudyID'] = ds.StudyID
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
        patient=patient,
        ct=ct,
        flip_angle=flip_angle,
        repetition_time=repetition_time,
        acquisition_time=acquisition_time,
        max_base=max_base,
        center_curve=center_curve,
        center_grad=center_grad,
        metadata=metadata)


def read_dce_dicoms(folder):
    dicoms = sorted([osp.join(folder, i) for i in os.listdir(folder) if i.endswith('dcm') or i.endswith('dicom')])
    flip_angle = []
    repetition_time = []
    acquisition_time = []
    data = []
    for d in tqdm(dicoms):
        ds = pydicom.dcmread(d)
        data.append(ds.pixel_array)
        acquisition_time.append(60 * (float(ds.AcquisitionTime[0:2]) * 60 + float(ds.AcquisitionTime[2:4])) + float(ds.AcquisitionTime[4:]))
        flip_angle.append(float(ds.FlipAngle))
        repetition_time.append(float(ds.RepetitionTime))
    acquisition_time = np.array(acquisition_time).reshape(75, 20)[:, 0]
    data = np.stack(data, axis=-1)
    h, w, _ = data.shape
    data = rearrange(data, 'h w (frames slices) -> h w slices frames', h=h, w=w, slices=20, frames=75)
    flip_angle = np.array(flip_angle)
    repetition_time = np.array(repetition_time)
    return dict(data=data, flip_angle=flip_angle, repetition_time=repetition_time, acquisition_time=acquisition_time)


def find_dce_folders(path):
    assert osp.isdir(path)
    patients = [osp.join(path, i) for i in os.listdir(path) if osp.isdir(osp.join(path, i))]
    exps = []
    for p in patients:
        exps1 = [osp.join(p, i) for i in os.listdir(p) if osp.isdir(osp.join(p, i))]
        exps.extend(exps1)
    dce_folders = []
    for exp in exps:
        candidates = [osp.join(exp, i) for i in os.listdir(exp) if 'iCAD-MCC_' in i or 'DCAD-MCC-DYN' in i or 't1_twist_tra' in i or 'Twist_dynamic' in i]
        if len(candidates) > 1:
            if len(candidates) == 2:
                if any(['MCC' in i for i in candidates]):
                    dce_folder = list(filter(lambda x: 'MCC' in x, candidates))[0]
            else:
                dce_folder = candidates[0]
        elif len(candidates) == 1:
            dce_folder = candidates[0]
        else:
            continue

        dce_folders.append(osp.abspath(dce_folder))
    return dce_folders


def find_ktrans_folder(path):
    path = osp.abspath(path)
    assert osp.isdir(path)
    candidates = [i for i in os.listdir(path) if osp.isdir(osp.join(path, i))]
    candidates = [osp.join(path, i) for i in os.listdir(path) if 'Ktrans' in i and osp.isdir(osp.join(path, i))]
    # candidates = [i for i in candidates if 'CLR' not in i]
    if len(candidates) > 1:
        if len([i for i in candidates if 'MCC' in i]) > 0:
            return [i for i in candidates if 'MCC' in i][0]
        else:
            print('??')
    else:
        warnings.warn('%s: cannot find Ktrans image.' % path)
        return None


def find_t2_folder(path):
    path = osp.abspath(path)
    assert osp.isdir(path)
    candidates = [i for i in os.listdir(path) if osp.isdir(osp.join(path, i))]
    candidates = [i for i in candidates if 't2' in i or 'T2' in i]
    if len(candidates) > 1:
        if len([i for i in candidates if 'tse_tra' in i]):
            candidates = [i for i in candidates if 'tse_tra' in i]
        elif True:
            pass
    candidates = [osp.join(path, i) for i in candidates]
    if len(candidates) >= 1:
        return candidates[0]
    else:
        warnings.warn('%s: cannot find T2 image.' % path)
        return None


if __name__ == '__main__':
    dce_folders = find_dce_folders('../dicom/')
    print(dce_folders)