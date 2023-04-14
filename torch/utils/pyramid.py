import torch
from torchvision.transforms._functional_tensor import _get_gaussian_kernel1d


def pyramid1d(x, sigmas=(1, 2, 3)):
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 3
    n, _, d = x.shape
    pyramid = []
    for s in sigmas:
        pyramid.append(gaussian_filter1d(x, s))
    return torch.cat(pyramid, dim=1)

def gaussian_filter1d(x, sigma, truncate=4):
    # x: [n c h]
    assert x.ndim == 3
    dim = x.shape[1]
    kernel_size = int(round((sigma * truncate + 1) // 2) * 2 + 1)
    kernel = _get_gaussian_kernel1d(
        kernel_size=kernel_size,
        sigma=sigma).view(1, 1, -1)
    kernel = kernel.expand(dim, 1, kernel.size(-1))
    padding = (kernel_size - 1) // 2
    x = torch.nn.functional.pad(x, pad=[padding, padding],  mode='replicate')
    y = torch.nn.functional.conv1d(x, kernel, groups=dim)
    return y