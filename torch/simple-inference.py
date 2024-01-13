
import torch
from prettytable import PrettyTable
from vlkit.medical import read_dicoms
from vlkit.lrscheduler import MultiStepScheduler
from vlkit.utils import   get_logger
import time, sys, argparse
import os.path as osp
from utils.utils import save_slices_to_dicom, str2int
from aif import get_aif, interp_aif
from pharmacokinetic import process_patient, evaluate_curve
from utils.load_dce import load_dce_data


def parse_args():
    parser = argparse.ArgumentParser(description='DCE inference')
    parser.add_argument('data', help='data path')
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--max-lr', type=float, default=1e-2)
    parser.add_argument('--save-path', type=str, default='/tmp/dce-mri')
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
    save2dicom(results['ktrans'], save_dir=f'{saveto}/', example_dicom=example_dcm, description=f'Ktrans-{aif}-AIF-{method}')
    if 'beta' in results and results['beta'] is not None:
        save2dicom(results['beta'], save_dir=f'{saveto}/', example_dicom=example_dcm, description=f'beta-{method}')
        if aif == 'mixed':
            ktrans_x_beta = results['ktrans'] * results['beta']
            save2dicom(ktrans_x_beta, save_dir=f'{saveto}/', example_dicom=example_dcm, description=f'Ktrans-x-beta-{method}')
            #
            ktrans_x_beta = results['ktrans'] * results['beta'].exp()
            save2dicom(ktrans_x_beta, save_dir=f'{saveto}/', example_dicom=example_dcm, description=f'Ktrans-x-betaexp-{method}')


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


def process_patient(dce_data: dict, aif: str, max_iter=100, max_lr=1e-2, min_lr=1e-5):
    if not isinstance(dce_data, dict):
        raise RuntimeError(type(dce_data))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ct = dce_data['ct']
    h, w, slices, frames = ct.shape
    mask = ct.max(dim=-1).values >= ct.max(dim=-1).values.max() / 50
    N = mask.sum()
    data = ct[mask]

    shape = dce_data['ct'].shape[:-1]
    ktrans = torch.rand(data.size(0)).to(device)
    kep = torch.rand(data.size(0)).to(ktrans)
    t0 = torch.rand(data.size(0)).to(ktrans)
    if aif == 'mixed':
        beta = torch.rand(data.size(0)).to(ktrans)

    if aif == 'mixed':
        weinmann_aif, aif_t = get_aif(aif='weinmann', max_base=6, acquisition_time=dce_data['acquisition_time'], device=device)
        parker_aif, _ = get_aif(aif='parker', max_base=6, acquisition_time=dce_data['acquisition_time'], device=device)
        beta = beta.squeeze(-1)
        aif_cp = interp_aif(parker_aif, weinmann_aif, beta=beta)
    else:
        aif_cp, aif_t = get_aif(aif=aif, max_base=6, acquisition_time=dce_data['acquisition_time'], device=device)

    logger.info("Start iterative refinement.")
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------------------------- #
    ktrans = ktrans.to(device=device).requires_grad_()
    kep = kep.to(device=device).requires_grad_()
    t0 = t0.to(device=device).requires_grad_()

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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.save_path, f'infeerence_{timestamp}.log')
    logger = get_logger(name="DCE (inference)", log_file=log_file)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    try:
            dce_data = load_dce_data(args.data, device=device)
    except Exception as e:
        logger.warn(f"{args.data}: cannot load DCE data from {args.data}: {e}")
        logger.error(e)
        sys.exit(1)

    table_data = {}
    table = PrettyTable()
    for aif in ['mixed', "parker", "weinmann"]:
        logger.info(f"Process {args.data} with {aif} AIF")
        example_dcm = read_dicoms(args.data)[:dce_data['ct'].shape[-2]]
        # get our results
        results = process_patient(dce_data, aif=aif, max_iter=args.max_iter, max_lr=args.max_lr)
        table_data[f'{aif}'] = results['loss']

        if results is None:
            logger.warn(f'{args.data}@{aif} AIF: result is empty.')
            continue
        save_resuls_to_dicoms(results=results, aif=aif, method='Kai', saveto=args.save_path, example_dcm=example_dcm)
        logger.info(f"Results have been saved to {args.save_path}")

    for k, v in table_data.items():
        table.add_column(k, [v])
    logger.info(table)
