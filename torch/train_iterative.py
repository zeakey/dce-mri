import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, sys, argparse, time, math
from tqdm import tqdm
import vlkit.plt as vlplt

from mmcv.cnn import MODELS
from mmcv.utils import Config, DictAction, get_logger
from mmcv.runner import build_optimizer, set_random_seed

from tofts import tofts
from pharmacokinetic import (
    find_max_base,
    np2torch,
    calculate_reconstruction_loss,
    evaluate_curve,
    compare_results
)
from utils import (
    save_slices_to_dicom,
    inference,
    ct_loss,
)
from aif import get_aif
import load_dce
from aif import parker_aif, biexp_aif
from ct_sampler import CTSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-iters', type=lambda x: int(float(x)), default=5e4)
    parser.add_argument('--max-lr', type=float, default=1e-4)
    parser.add_argument('--log-freq', type=int, default=100)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--work-dir', type=str)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    args.max_iters = int(args.max_iters)
    args.log_freq = int(args.log_freq)
    return args


def train(model, optimizer, lrscheduler):
    pass


if __name__ == '__main__':
    from scipy.io import loadmat, savemat
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from vlkit.lrscheduler import CosineScheduler
    import os.path as osp

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # model
    model = MODELS.build(cfg.model).to(device)
    if args.pretrained is not None:
        model.load_state_dict(torch.load(args.pretrained))

    # read data
    dce_data = load_dce.load_dce_data('../dicom/10042_1_1Wnr0444/20161209/iCAD-MCC_33000', device=device)
    max_base = dce_data['max_base'].item()
    acquisition_time = dce_data['acquisition_time']
    matlab_results = loadmat('../tmp/10042_1_1Wnr0444-20161209.mat')

    aif_cp, aif_t = get_aif(cfg.aif, acquisition_time, max_base, device=device)

    if args.test is not None:
        assert osp.isdir(args.test)
        dce_folders = load_dce.load_dce_folders(args.test)
        for folder in dce_folders:
            dce_data = load_dce.load_dce_data(folder)
            max_base = dce_data['max_base'].item()
            acquisition_time = dce_data['acquisition_time']
            ct = dce_data['ct'].to(device)
            ktrans, kep, t0 = inference(model, ct)
            loss = calculate_reconstruction_loss(ktrans, kep, t0, ct, acquisition_time, aif_t, aif_cp)

            shared_kwargs = dict(PatientID=dce_data['patient'].upper(), PatientName=dce_data['patient'].upper())
            shared_kwargs.update(dce_data['metadata'])
            save_slices_to_dicom(ktrans.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom', dce_data['patient'], 'Parker-ktrans'), SeriesDescription='Parker-ktrans', SeriesNumber=30004, SeriesInstanceUID='30004',**shared_kwargs)
            save_slices_to_dicom(kep.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom', dce_data['patient'], 'Parker-kep'), SeriesDescription='Parker-kep', SeriesNumber=30005, SeriesInstanceUID='30005', **shared_kwargs)
            save_slices_to_dicom(t0.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom', dce_data['patient'], 'Parker-t0'), SeriesDescription='Parker-t0', SeriesNumber=30006, SeriesInstanceUID='30006', **shared_kwargs)
            save_slices_to_dicom(loss.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom', dce_data['patient'], 'Parker-loss'), SeriesDescription='Parker-loss', SeriesNumber=30007, SeriesInstanceUID='30007', **shared_kwargs)
        sys.exit()

        ktrans, kep, t0 = inference(model, dce_data['ct'])
        loss = calculate_reconstruction_loss(ktrans, kep, t0, dce_data['ct'], acquisition_time, aif_t, aif_cp)

        shared_kwargs = dict(PatientName='10042_1_1Wnr0444', PatientID='10042_1_1Wnr0444'.upper())

        save_slices_to_dicom(matlab_results['ktrans'], dicom_dir=osp.join(cfg.work_dir, 'dicom/matlab-ktrans/'), SeriesDescription='matlab-ktrans', SeriesNumber=30001, **shared_kwargs)
        save_slices_to_dicom(matlab_results['kep'], dicom_dir=osp.join(cfg.work_dir, 'dicom/matlab-kep/'), SeriesDescription='matlab-kep', SeriesNumber=30002, **shared_kwargs)
        save_slices_to_dicom(matlab_results['t0'], dicom_dir=osp.join(cfg.work_dir, 'dicom/matlab-t0/'), SeriesDescription='matlab-t0', SeriesNumber=30003, **shared_kwargs)
        #
        save_slices_to_dicom(ktrans.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/Parker-ktrans/'), SeriesDescription='Parker-ktrans', SeriesNumber=30004, **shared_kwargs)
        save_slices_to_dicom(kep.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/Parker-kep/'), SeriesDescription='Parker-kep', SeriesNumber=30005, **shared_kwargs)
        save_slices_to_dicom(t0.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/Parker-t0/'), SeriesDescription='Parker-t0', SeriesNumber=30006, **shared_kwargs)
        save_slices_to_dicom(loss.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/Parker-loss/'), SeriesDescription='Parker-loss', SeriesNumber=30007, **shared_kwargs)
        #
        save_slices_to_dicom(ktrans.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/Weinmann-ktrans/'), SeriesDescription='Weinmann-ktrans', SeriesNumber=30008, **shared_kwargs)
        save_slices_to_dicom(kep.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/Weinmann-kep/'), SeriesDescription='Weinmann-kep', SeriesNumber=30009, **shared_kwargs)
        save_slices_to_dicom(t0.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/Weinmann-t0/'), SeriesDescription='Weinmann-t0', SeriesNumber=300010, **shared_kwargs)
        save_slices_to_dicom(loss.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/Weinmann-loss/'), SeriesDescription='Weinmann-loss', SeriesNumber=30011, **shared_kwargs)
        #
        save_slices_to_dicom(ktrans.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/FH-ktrans/'), SeriesDescription='FH-ktrans', SeriesNumber=30012, **shared_kwargs)
        save_slices_to_dicom(kep.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/FH-kep/'), SeriesDescription='FH-kep', SeriesNumber=30013, **shared_kwargs)
        save_slices_to_dicom(t0.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/FH-t0/'), SeriesDescription='FH-t0', SeriesNumber=30014, **shared_kwargs)
        save_slices_to_dicom(loss.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, 'dicom/FH-loss/'), SeriesDescription='FH-loss', SeriesNumber=30015, **shared_kwargs)
        sys.exit()

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_logger(name="DCE", log_file=log_file)
    logger.info(cfg)

    logger.info("Setting random seed to %d" % args.seed)
    set_random_seed(args.seed)

    writer = SummaryWriter(osp.join(cfg.work_dir, 'tensorboard'))

    batch_size = 256

    optimizer = build_optimizer(model, cfg.optimizer)
    lrscheduler = CosineScheduler(epoch_iters=args.max_iters, epochs=1, max_lr=args.max_lr, min_lr=args.max_lr*1e-5, warmup_iters=20)

    param_sampler = CTSampler(cfg.sampler)

    for i in range(args.max_iters):
        params_ref = param_sampler.sample(batch_size).to(device)
        ktrans_ref, kep_ref, t0_ref = params_ref.chunk(dim=1, chunks=3)

        # ct reference
        ct_reference_clean = evaluate_curve(ktrans_ref, kep_ref, t0_ref, aif_t, aif_cp, t=acquisition_time)
        noise = torch.randn(ct_reference_clean.shape, device=device) * torch.rand(ct_reference_clean.shape, device=device) / 2
        ct_reference = ct_reference_clean * (1 + noise)
        ct_reference[ct_reference < 0] = 0


        if np.random.uniform() > 0.5:
            noise = torch.randn(ktrans_ref.shape, device=ktrans_ref.device) / 4
        else:
            noise = 0
        ktrans = ktrans_ref + noise
        
        if np.random.uniform() > 0.5:
            noise = torch.randn(ktrans_ref.shape, device=ktrans_ref.device) * 2.5
        else:
            noise = 0
        kep = kep_ref + noise

        if np.random.uniform() > 0.5:
            noise = torch.randn(ktrans_ref.shape, device=ktrans_ref.device) * np.random.uniform(0, 1/3)
        else:
            noise = 0
        t0 = t0_ref + noise

        ktrans = ktrans.clamp(0, 1)
        kep = kep.clamp(0, 10)
        t0 = t0.clamp(0, 0.25)

        params = torch.cat((ktrans, kep, t0), dim=1)

        ct = evaluate_curve(ktrans, kep, t0, aif_t, aif_cp, t=acquisition_time)

        output = model(torch.cat((ct, ct_reference), dim=1).transpose(dim0=1, dim1=2))
        output = torch.cat(output, dim=1)
        target = params_ref - params

        params_pred = params + output
        ktrans_pred, kep_pred, t0_pred = params_pred.chunk(dim=1, chunks=3)
        ct_pred = evaluate_curve(ktrans_pred, kep_pred, t0_pred, aif_t, aif_cp, t=acquisition_time)

        if hasattr(cfg, 'debug') and cfg.debug and i % 500 == 0:
            n = 10
            cols = 5
            rows = int(math.ceil(n // cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
            indice = np.random.choice(ct.shape[0], n)
            ct_ = ct.cpu().numpy()
            ct_reference_ = ct_reference.cpu().numpy()
            ct_reference_clean_ = ct_reference_clean.cpu().numpy()
            ct_pred_ = ct_pred.detach().cpu().numpy()

            for j, k in enumerate(indice):
                r, c = j // cols, j % cols
                axes[r, c].plot(ct_reference_[k, :].flatten(), color='k')
                axes[r, c].plot(ct_reference_clean_[k, :].flatten(), '--', color='k')
                axes[r, c].plot(ct_[k, :].flatten(), color='tab:red')
                axes[r, c].plot(ct_pred_[k, :].flatten(), color='tab:blue')
                axes[r, c].grid()
                axes[r, c].legend(['CT-ref', 'CT-ref (clean)', 'CT', 'CT-pred'], loc='lower right')
                axes[r, c].set_title('CT %.3f, %.3f, %.3f \n CT_ref %.3f, %.3f, %.3f \n CT_pred %.3f, %.3f, %.3f' % (ktrans[k].item(), kep[k].item(), t0[k].item(), ktrans_ref[k].item(), kep_ref[k].item(), t0_ref[k].item(), ktrans_pred[k].item(), kep_pred[k].item(), t0_pred[k].item()))
            plt.tight_layout()
            fn = osp.join(cfg.work_dir, 'debug', 'curves-iter%d.jpg' % i)
            os.makedirs(osp.dirname(fn), exist_ok=True)
            plt.savefig(fn)

        if cfg.loss == 'mse':
            loss_func = torch.nn.functional.mse_loss
        elif cfg.loss == 'l1':
            loss_func = torch.nn.functional.l1_loss
        else:
            raise TypeError(cfg.loss)

        loss_param = loss_func(output, target)
        loss_ct = ct_loss(ct_pred, ct_reference, loss_func=loss_func)

        loss = loss_param + loss_ct

        optimizer.zero_grad()
        loss.backward()

        lr = lrscheduler.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if i % args.log_freq == 0 or i == args.max_iters - 1 or i == 0:
            logger.info(
                "iter {iter} loss={loss} (param={loss_param}, ct={loss_ct}) lr={lr} Ktrans (min={ktrans_min:.2f}, max={ktrans_max:.2f}, mean={ktrans_mean:.2f}), Kep (min={kep_min:.2f}, max={kep_max:.2f}, mean={kep_mean:.2f}), t0 (min={t0_min:.2f}, max={t0_max:.2f}, mean={t0_mean:.2f}) ".format(
                    iter='%.6d' % i,
                    loss='%.1e' % loss.item(), loss_param='%.1e' % loss_param.item(),  loss_ct='%.1e' % loss_ct.item(),
                    lr='%.3e' % lr,
                    ktrans_min=ktrans.min().item(), ktrans_max=ktrans.max().item(), ktrans_mean=ktrans.mean().item(),
                    kep_min=kep.min().item(), kep_max=kep.max().item(), kep_mean=kep.mean().item(),
                    t0_min=t0.min().item(), t0_max=t0.max().item(), t0_mean=t0.mean().item(),
                ))

            writer.add_scalar('train/lr', lr, i)
            writer.add_scalar('train/loss', loss.mean().item(), i)
            writer.add_scalar('train/loss-param', loss_param.mean().item(), i)
            writer.add_scalar('train/loss-ct', loss_ct.mean().item(), i)

        if (i + 1) % 1e4 == 0:
            torch.save(
                model.state_dict(),
                osp.join(cfg.work_dir, 'model-iter%.5d.pth' % (i + 1))
            )

    # test on a patient
    # dce_data = load_dce('../../dicom/10042_1_1Wnr0444/20161209/iCAD-MCC_33000/')
    data = loadmat('../tmp/10042_1_1Wnr0444-20161209.mat')
    data = np2torch(data)
    ct = data['dce_ct']
    ktrans, kep, t0 = inference(model, ct.to(device))
    ktrans = ktrans.cpu()
    matlab_ktrans = data['ktrans']

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    vlplt.clear_ticks(axes)

    for idx, i in enumerate(range(5, 15)):
        axes[0, idx].imshow(matlab_ktrans[70:110, 70:100, i])
        axes[0, idx].set_title('Slice#%.2d' % i)
        if idx == 0:
            axes[0, idx].set_ylabel('Matlab (Prostate)')
        #
        axes[1, idx].imshow(ktrans[70:110, 70:100, i])
        if idx == 0:
            axes[1, idx].set_ylabel('Torch (Prostate)')
    plt.tight_layout()
    writer.add_figure('ktrans', fig)
    writer.close()

