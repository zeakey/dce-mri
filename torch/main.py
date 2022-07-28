import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, sys, argparse, time
from einops import rearrange
from tqdm import tqdm
import vlkit.plt as vlplt

from mmcv.cnn import MODELS
from mmcv.utils import Config, DictAction, get_logger
from mmcv.runner import build_optimizer, set_random_seed

from pharmacokinetic import (
     tofts, np2torch,
     compare_results)
from utils import (
    read_dce_folder,
    find_max_base,
    save_slices_to_dicom,
    load_dce
)
from aif import parker_aif, biexp_aif
from ct_sampler import CTSampler, generate_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-iters', type=lambda x: int(float(x)), default=5e4)
    parser.add_argument('--max-lr', type=float, default=1e-4)
    parser.add_argument('--log-freq', type=int, default=100)
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--work-dir', type=str)
    parser.add_argument('--pretrained', type=str, default='./')
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



def inference(model, data):
    assert data.ndim == 4
    model.eval()

    h, w, s, f = data.shape
    data = rearrange(data, 'h w s f -> h (w s) f')
    ktrans = torch.zeros(h, w*s).to(data)
    kep = torch.zeros(h, w*s).to(data)
    t0 = torch.zeros(h, w*s).to(data)

    print('Start inference...')
    tic = time.time()
    with torch.no_grad():
        for i, d in enumerate(tqdm(data)):
            ktrans_, kep_, t0_ = model(d.unsqueeze(dim=-1))
            ktrans[i, ] = ktrans_.squeeze(dim=1)
            kep[i, ] = kep_.squeeze(dim=1)
            t0[i, ] = t0_.squeeze(dim=1)
    toc = time.time()
    print('Done, %.3fs elapsed.' % (toc-tic))
    ktrans = rearrange(ktrans, 'h (w s) -> h w s', w=w)
    kep = rearrange(kep, 'h (w s) -> h w s', w=w)
    t0 = rearrange(t0, 'h (w s) -> h w s', w=w)
    return ktrans, kep, t0



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

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_logger(name="DCE", log_file=log_file)
    logger.info(cfg)

    logger.info("Setting random seed to %d" % args.seed)
    set_random_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = MODELS.build(cfg.model)
    model = model.to(device)
    logger.info('Model:\n', model)

    writer = SummaryWriter(osp.join(cfg.work_dir, 'tensorboard'))

    batch_size = 256

    optimizer = build_optimizer(model, cfg.optimizer)
    logger.info(optimizer)

    lrscheduler = CosineScheduler(epoch_iters=args.max_iters, epochs=1, max_lr=args.max_lr, min_lr=args.max_lr*1e-5, warmup_iters=20)

    # read data
    dicom_data = read_dce_folder('../dicom/10042_1_1Wnr0444/20161209/iCAD-MCC_33000')
    dce_data = torch.tensor(dicom_data['data'].astype(np.float32)).float().to(device=device)
    acquisition_time = torch.tensor(dicom_data['acquisition_time']).to(dce_data)
    acquisition_time = acquisition_time - acquisition_time[0]
    repetition_time = torch.tensor(dicom_data['repetition_time']).to(dce_data)
    flip_angle = torch.tensor(dicom_data['flip_angle']).to(dce_data)

    max_base = find_max_base(dce_data)

    # second to minite
    acquisition_time = acquisition_time / 60

    # we assume 7 minutes aif at most
    aif_t = torch.arange(0, acquisition_time[-1], 1/60, dtype=torch.float64).to(acquisition_time)
    aif_t = aif_t - (acquisition_time[max_base] / (1 / 60)).ceil() * (1 / 60)
    hct = 0.42
    if cfg.aif == 'parker':
        aif_cp = parker_aif(
            a1=0.809,
            a2=0.330,
            t1=0.17046,
            t2=0.365,
            sigma1=0.0563,
            sigma2=0.132,
            alpha=1.050,
            beta=0.1685,
            s=38.078,
            tau=0.483,
            t=acquisition_time) / (1 - hct)
    elif cfg.aif == 'weinmann':
        aif_cp = biexp_aif(3.99, 4.78, 0.144, 0.011, aif_t) / (1 - hct)
    elif cfg.aif == 'fh':
        aif_cp = biexp_aif(24, 6.2, 3.0, 0.016, aif_t) / (1 - hct)
    else:
        raise ValueError('Invalid AIF: %s' % cfg.aif)

    aif_t = aif_t.view(1, 1, -1).repeat(batch_size, 1, 1)
    aif_cp = aif_cp.view(1, 1, -1).repeat(batch_size, 1, 1)

    param_sampler = CTSampler()

    for i in range(args.max_iters):
        params = param_sampler.sample(batch_size).to(dce_data)
        ktrans, kep, t0 = params.chunk(dim=1, chunks=3)

        ct = generate_data(ktrans, kep, t0, aif_t, aif_cp, t=acquisition_time)

        # output: (ktrans, kep, t0)
        output = model(ct.transpose(1, 2), acquisition_time)
        assert len(output) == 3

        output = torch.cat(output, dim=1)
        if cfg.loss == 'mse':
            loss = torch.nn.functional.mse_loss(output, params)
        elif cfg.loss == 'l1':
            loss = torch.nn.functional.l1_loss(output, params)
        elif cfg.loss == 'mixed':
            loss = (torch.nn.functional.l1_loss(output, params) + torch.nn.functional.mse_loss(output, params)) / 2
        else:
            raise TypeError(cfg.loss)

        optimizer.zero_grad()
        loss.backward()

        lr = lrscheduler.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if i % args.log_freq == 0 or i == args.max_iters - 1 or i == 0:
            logger.info(
                "iter {iter} loss={loss}  lr={lr} Ktrans (min={ktrans_min:.2f}, max={ktrans_max:.2f}, mean={ktrans_mean:.2f}), Kep (min={kep_min:.2f}, max={kep_max:.2f}, mean={kep_mean:.2f}), t0 (min={t0_min:.2f}, max={t0_max:.2f}, mean={t0_mean:.2f}) ".format(
                    iter='%.6d' % i, loss='%.1e' % loss.item(), lr='%.3e' % lr,
                    ktrans_min=ktrans.min().item(), ktrans_max=ktrans.max().item(), ktrans_mean=ktrans.mean().item(),
                    kep_min=kep.min().item(), kep_max=kep.max().item(), kep_mean=kep.mean().item(),
                    t0_min=t0.min().item(), t0_max=t0.max().item(), t0_mean=t0.mean().item(),
                ))
            writer.add_scalar('lr', lr, i)
            writer.add_scalar('loss', loss.mean().item(), i)

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

