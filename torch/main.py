from multiprocessing.sharedctypes import Value
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
    save_slices_to_dicom
)
from aif import parker_aif, biexp_aif
from ct_sampler import CTSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-iters', type=lambda x: int(float(x)), default=1e5)
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


def generate_data(ktrans, kep, t0, aif_t, aif_cp, t):
    batch_size = ktrans.shape[0]
    t = t.view(1, 1, -1).repeat(batch_size, 1, 1)
    ct = tofts(ktrans, kep, t0, t, aif_t, aif_cp)
    noice = torch.randn(ct.shape, device=ct.device) / 4
    ct += ct * noice
    ct[ct < 0] = 0
    return ct


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

    if args.test:
        assert osp.isfile(args.pretrained)
        pretrained = torch.load(args.pretrained)
        model.load_state_dict(pretrained)
        model.eval()

        data = loadmat('../tmp/patient-0.mat')
        ct = torch.from_numpy(data['dce_ct']).to(device).float()
        t = torch.from_numpy(data['time_dce']).to(device).float() / 60

        h, w, s, f = ct.shape
        ct = rearrange(ct, 'h w s f -> h (w s) f')
        ktrans = torch.zeros(h, w*s).to(ct)
        kep = torch.zeros(h, w*s).to(ct)
        t0 = torch.zeros(h, w*s).to(ct)

        tic = time.time()
        with torch.no_grad():
            for i, d in enumerate(tqdm(ct)):
                ktrans_, kep_, t0_ = model(d.unsqueeze(dim=-1), t)
                ktrans[i, ] = ktrans_.squeeze(dim=1)
                kep[i, ] = kep_.squeeze(dim=1)
                t0[i, ] = t0_.squeeze(dim=1)
        ct = rearrange(ct, 'h (w s) f -> h w s f', w=w)

        ktrans = rearrange(ktrans, 'h (w s) -> h w s', w=w)
        kep = rearrange(kep, 'h (w s) -> h w s', w=w)
        t0 = rearrange(t0, 'h (w s) -> h w s', w=w)
        print("ETA: %f" % (time.time() - tic))

        ktrans = ktrans.cpu().numpy()
        kep = kep.cpu().numpy()
        t0 = t0.cpu().numpy()

        save_slices_to_dicom(ktrans, dicom_dir=osp.join(cfg.work_dir, 'dicom/tansformer-ktrans/'), SeriesDescription='Transformer-ktrans', SeriesNumber=30000)
        save_slices_to_dicom(kep, dicom_dir=osp.join(cfg.work_dir, 'dicom/tansformer-kep/'), SeriesDescription='Transformer-kep', SeriesNumber=30001)
        save_slices_to_dicom(t0, dicom_dir=osp.join(cfg.work_dir, 'dicom/tansformer-t0/'), SeriesDescription='Transformer-t0', SeriesNumber=30002)
        #
        save_slices_to_dicom(data['ktrans'], dicom_dir=osp.join(cfg.work_dir, 'dicom/Matlab-ktrans/'), SeriesDescription='Matlab-ktrans', SeriesNumber=30003)
        save_slices_to_dicom(data['kep'], dicom_dir=osp.join(cfg.work_dir, 'dicom/Matlab-kep/'), SeriesDescription='Matlab-kep', SeriesNumber=30004)
        save_slices_to_dicom(data['t0'], dicom_dir=osp.join(cfg.work_dir, 'dicom/Matlab-t0/'), SeriesDescription='Matlab-t0', SeriesNumber=30005)

        compare_results(ktrans, data['ktrans'], name1='Transformer', name2='Matlab', fig_filename=osp.join(cfg.work_dir, 'ktrans.pdf'))
        compare_results(kep, data['kep'], name1='Transformer', name2='Matlab', fig_filename=osp.join(cfg.work_dir, 'kep.pdf'))
        compare_results(t0, data['t0'], name1='Transformer', name2='Matlab', fig_filename=osp.join(cfg.work_dir, 't0.pdf'))
        sys.exit()


    tensorboard = SummaryWriter(osp.join(cfg.work_dir, 'tensorboard'))

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

    shift_base = False
    if shift_base:
        # important! shift max_base for uniform ct
        dce_data = dce_data[:, :, :, max_base:]
        dce_data = torch.cat((
                dce_data,
                dce_data[:, :, :, -1:].repeat(1, 1, 1, max_base)
            ), dim=-1)

        acquisition_time = acquisition_time[max_base:]
        interval = (acquisition_time[-1] - acquisition_time[0]) / (acquisition_time.numel() - 1)

        acquisition_time = torch.cat((acquisition_time, acquisition_time[-1] + torch.arange(1, max_base+1).to(device) * interval), dim=0)
        acquisition_time = acquisition_time - acquisition_time[0]

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

        ktrans, kep, t0 = model(ct.transpose(1, 2), acquisition_time)

        output = torch.cat((ktrans, kep, t0), dim=1)
        loss = torch.nn.functional.l1_loss(output, params)

        optimizer.zero_grad()
        loss.backward()

        lr = lrscheduler.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if i % args.log_freq == 0 or i == args.max_iters - 1 or i == 0:
            logger.info("iter %.3d loss=%.3f  lr=%.2e" % (i, loss.item(), lr))
            tensorboard.add_scalar('lr', lr, i)
            tensorboard.add_scalar('loss', loss.mean().item(), i)

        if i % 5e3 == 0:
            torch.save(
                model.state_dict(),
                osp.join(cfg.work_dir, 'model-iter%.5d.pth' % i)
            )

