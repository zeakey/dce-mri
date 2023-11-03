import torch
from torchvision.transforms._functional_tensor import _get_gaussian_kernel1d


def pyramid1d(x, sigmas=(1, 2, 3), return_kernel=False):
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 3
    pyramid = []
    kernels= []

    for s in sigmas:
        if return_kernel:
            y, k = gaussian_filter1d(x, s, return_kernel=return_kernel)
            pyramid.append(y)
            kernels.append(k)
        else:
            pyramid.append(gaussian_filter1d(x, s))
    pyramid = torch.cat(pyramid, dim=1)
    if return_kernel:
        return pyramid, kernels
    else:
        return pyramid


def gaussian_filter1d(x, sigma, truncate=4, return_kernel=False):
    # x: [n c h]
    assert x.ndim == 3
    dim = x.shape[1]
    kernel_size = int(round((sigma * truncate + 1) // 2) * 2 + 1)
    kernel = _get_gaussian_kernel1d(
        kernel_size=kernel_size,
        sigma=sigma).view(1, 1, -1)
    kernel = kernel.expand(dim, 1, kernel.size(-1)).to(x)
    padding = (kernel_size - 1) // 2
    x = torch.nn.functional.pad(x, pad=[padding, padding],  mode='replicate')
    y = torch.nn.functional.conv1d(x, kernel, groups=dim)
    if return_kernel:
        return y, kernel
    else:
        return y