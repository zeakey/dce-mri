import torch
from torch.utils.tensorboard import SummaryWriter
import os, sys, argparse, time
from einops import rearrange
from tqdm import tqdm
import vlkit.plt as vlplt

from mmcv.cnn import MODELS
from mmcv.utils import Config

from pharmacokinetic import (
    parker_aif,
     tofts, np2torch,
     save_slices_to_dicom,
     compare_results)
from ct_sampler import CTSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--max-iters', type=lambda x: int(float(x)), default=1e5)
    parser.add_argument('--max-lr', type=float, default=1e-4)
    parser.add_argument('--log-freq', type=int, default=100)
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--work-dir', type=str)
    parser.add_argument('--pretrained', type=str, default='./')
    args = parser.parse_args()
    args.max_iters = int(args.max_iters)
    args.log_freq = int(args.log_freq)
    return args


def train(model, optimizer, lrscheduler):
    pass


def generate_data(ktrans, kep, t0, aif_t, aif_cp):
    batch_size = ktrans.shape[0]
    # t = torch.rand(batch_size, 1, 75).sort(dim=2).values.to(aif_t) * aif_t.max()
    # t = torch.linspace(0, aif_t.max(), 75, device=aif_t.device).view(1, 1, -1).repeat(batch_size, 1, 1)
    t = time_dce.view(1, 1, -1).repeat(batch_size, 1, 1)
    ct = tofts(ktrans, kep, t0, t, aif_t, aif_cp)
    noice = torch.randn(ct.shape, device=ct.device) / 4
    ct += ct * noice
    return ct, t / aif_t.max()


if __name__ == '__main__':
    from scipy.io import loadmat, savemat
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from vlkit.lrscheduler import CosineScheduler
    import os.path as osp

    args = parse_args()

    cfg = Config.fromfile(args.config)

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
    print(cfg)

    tensorboard = SummaryWriter(osp.join(cfg.work_dir, 'tensorboard'))

    batch_size = 256

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)

    # optimizer stolen from https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/schedules/imagenet_bs1024_adamw_swin.py
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))

    lrscheduler = CosineScheduler(epoch_iters=args.max_iters, epochs=1, max_lr=args.max_lr, min_lr=args.max_lr*1e-5, warmup_iters=20)

    data = loadmat('../tmp/patient-0.mat')
    data = np2torch(data)

    time_dce = (data['time_dce'] / 60).to(device=device)
    aif_t = torch.arange(0, time_dce[-1].item(), 1/60, dtype=torch.float64).to(time_dce)

    hct = 0.42
    aif_cp = parker_aif(
        a1=0.809, a2=0.330,
        t1=0.17046, t2=0.365,
        sigma1=0.0563, sigma2=0.132,
        alpha=1.050, beta=0.1685, s=38.078, tau=0.483,
        t=aif_t - (time_dce[int(data['maxBase'].item())-1] * 60).ceil() / 60
        ) / (1 - hct)

    aif_t = aif_t.view(1, 1, -1).repeat(batch_size, 1, 1)
    aif_cp = aif_cp.view(1, 1, -1).repeat(batch_size, 1, 1)

    param_sampler = CTSampler()

    for i in range(args.max_iters):
        if False:
            ktrans = torch.zeros(batch_size, 1).uniform_(0, 2).to(aif_t)
            kep = torch.zeros(batch_size, 1).uniform_(0, 2).to(aif_t)
            t0 = torch.zeros(batch_size, 1).uniform_(0, 0.4).to(aif_t)

            params = torch.stack((ktrans, kep, t0)).squeeze(dim=-1).transpose(0, 1)
        else:
            params = param_sampler.sample(batch_size).to(aif_t)
            ktrans, kep, t0 = params.chunk(dim=1, chunks=3)

        ct, t = generate_data(ktrans, kep, t0, aif_t, aif_cp)

        t = t.squeeze(dim=1)

        ktrans, kep, t0 = model(ct.transpose(1, 2), t)

        # scale = [10, 1, 200]
        # scale = torch.tensor([10, 1, 200]).view(1, 3).to(ktrans)
        scale = torch.tensor([1, 1, 1]).view(1, 3).to(ktrans)
        output = torch.cat((ktrans, kep, t0), dim=1)
        loss = (torch.nn.functional.mse_loss(output, params, reduction='none') * scale).mean()
        # loss = torch.nn.functional.l1_loss(output, params)

        optimizer.zero_grad()
        loss.backward()

        lr = lrscheduler.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if i % args.log_freq == 0 or i == args.max_iters - 1 or i == 0:
            print("iter %.3d loss=%.3f  lr=%.2e" % (i, loss.item(), lr))
            tensorboard.add_scalar('lr', lr, i)
            tensorboard.add_scalar('loss', loss.mean().item(), i)

    torch.save(model.state_dict(), osp.join(cfg.work_dir, 'model.pth'))
        # plt.plot(curves[0, 0, :].cpu())
        # plt.savefig('curve.pdf')
