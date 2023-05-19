import torch
import torch.nn as nn
from mmcv.cnn import MODELS

from utils import pyramid1d

from collections import OrderedDict


@MODELS.register_module()
class DCETransformer(nn.Module):
    def __init__(
        self,
        seq_len,
        use_grad=False,
        pyramid_sigmas=None,
        num_fourier_features=0,
        num_layers=2,
        embed_dims=32,
        num_heads=2,
        output_keys=['ktrans', 'kep', 't0'],
        feedforward_channels=32,
        drop_rate=0
        ) -> None:

        super().__init__()
        self.num_outputs = len(output_keys)
        self.output_keys = output_keys
        self.use_grad = use_grad
        self.pyramid_sigmas = pyramid_sigmas
        self.num_fourier_features = num_fourier_features

        # input dimension
        if self.use_grad:
            input_dim = 2
        else:
            input_dim = 1
        if self.pyramid_sigmas is not None:
            input_dim *= len(self.pyramid_sigmas)
        if self.num_fourier_features > 1:
            input_dim *= self.num_fourier_features * 2
        # position embeding for start and end time step
        self.register_buffer('pos_embed', torch.randn(1, seq_len+1, embed_dims))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.linear = nn.Linear(input_dim, embed_dims)
        self.layers = nn.ModuleList()
        self.output_layer = nn.ModuleList()
        for _ in range(self.num_outputs):
            # use ReLU to prevent negative outputs
            self.output_layer.append(nn.Sequential(nn.Linear(embed_dims, 1), nn.ReLU()))
        for _ in range(num_layers):
            transformer = nn.TransformerEncoderLayer(
                    d_model=embed_dims,
                    nhead=num_heads,
                    dim_feedforward=feedforward_channels,
                    batch_first=True)
            self.layers.append(transformer)

        self.init_weights()

    def init_weights(self):
        for m in self.output_layer:
            nn.init.normal_(m[0].weight, std=0.01)

    def forward(self, x, t=None):
        B = x.shape[0]
        if self.use_grad:
            # gradient of series
            grad = torch.gradient(x, dim=1)[0]
            x = torch.cat((x, grad), dim=2)

        if self.pyramid_sigmas is not None:
            # [batch length dim]
            x = pyramid1d(x.transpose(1, 2), sigmas=self.pyramid_sigmas).transpose(1, 2)
        if self.num_fourier_features > 1:
            fourier_features = []
            for freq in range(self.num_fourier_features):
                fourier_features.append(
                    torch.sin(x * 2 * torch.pi * (freq + 1))
                )
                fourier_features.append(
                    torch.cos(x * 2 * torch.pi * (freq + 1))
                )
            x = torch.cat(fourier_features, dim=-1)
        x = self.linear(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = self.pos_embed + x
        x = self.drop_after_pos(x)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :]
        output = OrderedDict()
        for k, layer in zip(self.output_keys, self.output_layer):
            output[k] = layer(x)
        return output


if __name__ == '__main__':
    import time
    model = MODELS.build(dict(type='DCETransformer', seq_len=75, use_grad=False)).cuda()
    x = torch.randn(160*160, 75, 2).cuda()
    tic = time.time()
    print(model(x))
    print((time.time() - tic) * 20)
