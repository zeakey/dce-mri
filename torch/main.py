import torch
from models.rnn import RNNModel
from pharmacokinetic import parker_aif, tofts, np2torch


if __name__ == '__main__':
    from scipy.io import loadmat, savemat
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from vlkit.lrscheduler import CosineScheduler
    import os.path as osp
    
    work_dir = 'work_dirs/GRU'
    device = torch.device('cuda')
    batch_size = 256

    model = RNNModel(input_size=2, num_layers=12).to(device)
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

    max_iter = int(1e4)
    max_lr = 1e-2
    lrscheduler = CosineScheduler(epoch_iters=max_iter, epochs=1, max_lr=max_lr, min_lr=max_lr*1e-5, warmup_iters=20)

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

    for i in range(max_iter):
        ktrans = torch.zeros(batch_size, 1).uniform_(0, 5).to(aif_t)
        kep = torch.zeros(batch_size, 1).uniform_(0, 50).to(aif_t)
        t0 = torch.zeros(batch_size, 1).uniform_(0, 0.25).to(aif_t)

        params = torch.stack((ktrans, kep, t0)).squeeze(dim=-1).transpose(0, 1)
        curves = tofts(ktrans, kep, t0, aif_t, aif_t, aif_cp)
        grad = torch.gradient(curves.squeeze(dim=1), dim=1)[0].unsqueeze(dim=1)
        input = torch.cat((curves, grad), dim=1)

        ktrans, kep, t0 = model(input.transpose(1, 2))
        output = torch.cat((ktrans, kep, t0), dim=1)
        loss = torch.nn.functional.mse_loss(output, params)
        # loss = torch.nn.functional.l1_loss(output, params)

        optimizer.zero_grad()
        loss.backward()

        lr = lrscheduler.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if i % 10 == 0:
            print("iter %.3d loss=%.3f  lr=%.2e" % (i, loss.item(), lr))

    torch.save(model.state_dict(), osp.join(work_dir, 'model.pth'))
        # plt.plot(curves[0, 0, :].cpu())
        # plt.savefig('curve.pdf')