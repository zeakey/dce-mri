import torch
from einops import rearrange


def tofts(ktrans, kep, t0, t, aif_t, aif_cp):
    """
    Tofts model

    ktrans, kep, t0: [h, w]
    t: [h, w, t1]
    aif_t, aif_cp: [h, w, t2]
    For batch process, here I convert the individual convolution where each case has its own
    kernel to a grouped 1-D convolution.
    """
    assert ktrans.ndim == 2 and kep.ndim == 2 and t0.ndim == 2
    assert t.ndim == 3 and aif_t.ndim == 3, 't.shape=%s, aif_t.shape=%s' % (str(t.shape), str(aif_t.shape))
    h, w = ktrans.shape
    t1, t2 = t.shape[2], aif_t.shape[2]
    dt = aif_t[:, :, 1] - aif_t[:, :, 0]

    assert t.shape[:2] == torch.Size([h, w]), t.shape
    assert aif_t.shape[:2] == torch.Size([h, w]), aif_t.shape
    assert aif_cp.shape[:2] == torch.Size([h, w]), aif_cp.shape

    # impulse response
    impulse = ktrans.unsqueeze(dim=-1) * (-kep.unsqueeze(dim=-1) * aif_t).exp()

    # rearrange shapes for 1-D convolution
    # see https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html
    # for document to `torch.nn.functional.conv1d`
    aif_cp = rearrange(aif_cp, 'h w t2 -> 1 (h w) t2')
    impulse = rearrange(impulse, 'h w t2 -> (h w) 1 t2')
    # convolve the impulse response with AIF
    conv = torch.nn.functional.conv1d(input=aif_cp, weight=impulse.flip(dims=(-1,)), padding=t2-1, groups=h*w)[:, :, :t2]
    conv = rearrange(conv, '1 (h w) t2 -> h w t2', h=h, w=w) * dt.view(h, w, 1)

    # interpolate
    conv = rearrange(conv, 'h w t2 -> 1 1 (h w) t2', h=h, w=w, t2=t2)
    gridx = ((t - t0.view(h, w, 1)) / dt.view(h, w, 1))
    gridx = rearrange(gridx, 'h w t1 -> 1 (h w) t1 1')
    gridy = torch.arange(h*w).view(1, h*w, 1, 1).repeat(1, 1, t1, 1).to(gridx)
    # normalize to [-1, 1]
    gridx = gridx / (t2 - 1) * 2 -1
    if h * w == 1:
        gridy.fill_(-1)      
    elif h * w > 1:
        gridy = gridy / (h * w - 1) * 2 -1
    grid = torch.cat((gridx, gridy), dim=-1) # shape: [1, h*w, t1, 2]
    interp = torch.nn.functional.grid_sample(conv, grid, align_corners=True) # shape: [1, 1, h*w, t1]
    # matlab mean(interp) = 0.3483
    interp = rearrange(interp, '1 1 (h w) t1 -> h w t1', h=h, w=w, t1=t1)
    return interp


def tofts3d(ktrans, kep, t0, t, aif_t, aif_cp):
    """
    Tofts model on 3D volune
    """
    assert ktrans.ndim == 3
    assert ktrans.shape == kep.shape == t0.shape
    assert aif_t.ndim == 4
    assert aif_cp.shape == aif_t.shape
    h, w, slices = ktrans.shape
    ktrans = rearrange(ktrans, 'h w s -> h (w s)')
    kep = rearrange(kep, 'h w s -> h (w s)')
    t0 = rearrange(t0, 'h w s -> h (w s)')
    #
    t = rearrange(t, 'h w s t -> h (w s) t')
    aif_t = rearrange(aif_t, 'h w s t -> h (w s) t')
    aif_cp = rearrange(aif_cp, 'h w s t -> h (w s) t')

    ct = tofts(ktrans, kep, t0, t, aif_t, aif_cp)
    return rearrange(ct, 'h (w s) t -> h w s t', w=w, s=slices)