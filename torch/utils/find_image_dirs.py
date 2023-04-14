import os, warnings, re
from glob import glob
import os.path as osp
from datetime import datetime


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


def find_icad_ktrans(path):
    path = osp.abspath(path)
    assert osp.isdir(path)
    candidates = [i for i in os.listdir(path) if osp.isdir(osp.join(path, i))]
    candidates = [osp.join(path, i) for i in candidates if ('icad' in i.lower() or 'dcad' in i.lower()) and ('ktrans' in i.lower() or 'perm' in i.lower())]
    if len(candidates) >= 1 and any(['mcc' in c.lower() for c in candidates]):
        candidates = [c for c in candidates if 'mcc' in c.lower()]
    return candidates if len(candidates) > 0 else None


def find_histopathology(patient_id, exp_date):
    root = '/media/hdd1/IDX_Current/exported_annotations'
    subdirs = [s for s in os.listdir(root) if os.listdir(osp.join(root, s))]
    if patient_id in subdirs:
        hist_dir = osp.join(root, patient_id)
    elif patient_id.upper() in subdirs:
        hist_dir = osp.join(root, patient_id.upper())
    elif patient_id.lower() in subdirs:
        hist_dir = osp.join(root, patient_id.lower())
    else:
        warnings.warn(f"{patient_id} cannot find histopathology.")
        return None
    exp_date = datetime.strptime(exp_date, '%Y%m%d').strftime("%Y-%m-%d")

    if not osp.isdir(osp.join(hist_dir, exp_date)):
        warnings.warn(f"{osp.join(hist_dir, exp_date)} not exist.")
        candidates = glob(f"{hist_dir}/*-*-*/")
        hist_dir = candidates[0] if len(candidates) else None
    return hist_dir


if __name__ == '__main__':
    dce_folders = find_dce_folders('../dicom/')
    print(dce_folders)