import os, warnings
import os.path as osp


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