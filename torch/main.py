import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, sys, argparse, time, math
from einops import rearrange
from tqdm import tqdm
import vlkit.plt as vlplt
from collections import OrderedDict

from mmcv.cnn import MODELS
from mmcv.utils import Config, DictAction, get_logger
from mmcv.runner import build_optimizer, set_random_seed

from pharmacokinetic import (
    tofts,
    find_max_base,
    np2torch,
    calculate_reconstruction_loss,
    compare_results,
    evaluate_curve
)
from utils import (
    save_slices_to_dicom,
    inference,
    ct_loss,
    mix_l1_mse_loss
)
from aif import get_aif, interp_aif
from utils import load_dce
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

    if cfg.aif != 'mixed':
        aif_cp, aif_t = get_aif('parker', acquisition_time, max_base, device=device)
    else:
        parker_aif, aif_t = get_aif('parker', acquisition_time, max_base, device=device)
        weinmann_aif, _ = get_aif('weinmann', acquisition_time, max_base, device=device)
        fh_aif, _ = get_aif('fh', acquisition_time, max_base, device=device)
    # normalize AIFs

    if args.test is not None:
        assert osp.isdir(args.test)
        dce_folders = load_dce.find_dce_folders(args.test)
        for folder in dce_folders:
            dce_data = load_dce.load_dce_data(folder)
            max_base = dce_data['max_base'].item()
            acquisition_time = dce_data['acquisition_time'].to(device)
            ct = dce_data['ct'].to(device)
            ktrans, kep, t0 = inference(model, ct)
            loss = calculate_reconstruction_loss(ktrans, kep, t0, ct, acquisition_time, aif_t, aif_cp, relative=True)

            shared_kwargs = dict(PatientID=dce_data['patient'].upper(), PatientName=dce_data['patient'].upper())
            shared_kwargs.update(dce_data['metadata'])
            save_slices_to_dicom(ktrans.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, dce_data['patient'], 'Parker-ktrans'), SeriesDescription='Parker-ktrans', SeriesNumber=30004, SeriesInstanceUID='30004',**shared_kwargs)
            save_slices_to_dicom(kep.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, dce_data['patient'], 'Parker-kep'), SeriesDescription='Parker-kep', SeriesNumber=30005, SeriesInstanceUID='30005', **shared_kwargs)
            save_slices_to_dicom(t0.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, dce_data['patient'], 'Parker-t0'), SeriesDescription='Parker-t0', SeriesNumber=30006, SeriesInstanceUID='30006', **shared_kwargs)
            save_slices_to_dicom(loss.cpu().numpy(), dicom_dir=osp.join(cfg.work_dir, dce_data['patient'], 'Parker-loss'), SeriesDescription='Parker-loss', SeriesNumber=30007, SeriesInstanceUID='30007', **shared_kwargs)
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

    param_sampler = CTSampler(cfg.sampler, device=device)

    for i in range(args.max_iters):
        params = param_sampler.sample(batch_size)
        ktrans, kep, t0 = params["ktrans"], params["kep"], params["t0"]
        params_tensor = torch.cat((ktrans, kep, t0), dim=-1)
        if cfg.aif == 'mixed':
            aif_cp = interp_aif(parker_aif, weinmann_aif, beta=params['beta'])
            params_tensor = torch.cat((params_tensor, params['beta']), dim=-1)
        ct = evaluate_curve(ktrans, kep, t0, aif_t, aif_cp, t=acquisition_time)

        noise_scale = np.random.uniform(low=0, high=2, size=(batch_size))
        noise_scale = torch.from_numpy(noise_scale).to(ct).view(-1, 1)
        noise = torch.randn(ct.shape, device=ct.device) * noise_scale.view(-1, 1, 1)
        ct = ct + ct * noise
        ct[ct < 0] = 0
        params['noise_scale'] = noise_scale

        output = model(ct.transpose(dim0=1, dim1=2))
        output_params = torch.cat(list(output.values()), dim=-1)

        if cfg.aif == 'mixed':
            aif_recon = interp_aif(parker_aif, weinmann_aif, beta=output['beta'])
            ct_recon = evaluate_curve(output['ktrans'], output['kep'], output['t0'], aif_t=aif_t, aif_cp=aif_recon, t=acquisition_time)
        else:
            ct_recon = evaluate_curve(output['ktrans'], output['kep'], output['t0'], aif_t=aif_t, aif_cp=aif_cp, t=acquisition_time)

        if i % 100 == 0 and hasattr(cfg, 'debug') and cfg.debug:
            n = 20
            cols = 10
            rows = int(math.ceil(n // 10))
            fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
            indice = np.random.choice(ct.shape[0], n)
            ct_ = ct.cpu()
            ct_recon_ = ct_recon.cpu().detach()

            for ind, j in enumerate(indice):
                r, c = ind // cols, ind % cols
                axes[r, c].plot(ct_[j, :].flatten())
                axes[r, c].plot(ct_recon_[j, :].flatten())
                axes[r, c].grid()
                axes[r, c].set_title(
                    'noise: %.2f (%.2f)\n%.2f | %.2f | %.2f' % (output['noise_scale'][j].item(), noise_scale[j].item(), ktrans[j].item(), kep[j].item(), t0[j].item()),
                    fontsize=12)

            plt.tight_layout()
            fn = osp.join(cfg.work_dir, 'debug', 'curves-iter%d.jpg' % i)
            os.makedirs(osp.dirname(fn), exist_ok=True)
            plt.savefig(fn)
            plt.close()

        if cfg.loss == 'mse':
            loss_func = torch.nn.functional.mse_loss
        elif cfg.loss == 'l1':
            loss_func = torch.nn.functional.l1_loss
        elif cfg.loss == 'mixed':
            loss_func = mix_l1_mse_loss
        else:
            raise TypeError(cfg.loss)
        
        loss_param = OrderedDict()
        for k, v in output.items():
            loss_param[k] = loss_func(output[k], params[k])
        loss_ct = ct_loss(ct_recon, ct, loss_func=loss_func)
        loss = torch.stack(list(loss_param.values())).sum() + loss_ct

        optimizer.zero_grad()
        loss.backward()

        lr = lrscheduler.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if (i+1) % args.log_freq == 0 or (i +1)== args.max_iters or i == 0:
            log_str = f"[{i+1}|{args.max_iters}] loss={loss.item():.3f} l_ct={loss_ct.item():.3f} l_p={torch.stack(list(loss_param.values())).sum():.3f} ("
            for j1, (k, v) in enumerate(loss_param.items()):
                if j1 < len(loss_param) - 1:
                    log_str += f"{k}={v.item():.3f}, "
                else:
                    log_str += f"{k}={v.item():.3f}"
            log_str += f") lr={lr:.2e}, Ktrans({ktrans.min():.2f}, {ktrans.max():.2f}, {ktrans.mean():.2f}), Kep({kep.min():.2f}, {kep.max():.2f}, {kep.mean():.2f}), t0({t0.min():.2f}, {t0.max():.2f}, {t0.mean():.2f})"
            logger.info(log_str)

            writer.add_scalar('train/lr', lr, i)
            writer.add_scalar('train/loss', loss.item(), i)
            writer.add_scalar('train/loss-param', torch.stack(list(loss_param.values())).sum().item(), i)
            writer.add_scalar('train/loss-ct', loss_ct.mean().item(), i)
            for k, v in loss_param.items():
                writer.add_scalar(f'train/loss-param-{k}', v.item(), i)

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

