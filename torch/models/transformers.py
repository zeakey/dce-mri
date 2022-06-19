from turtle import forward
import torch
import torch.nn as nn
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer
from mmcv.cnn import MODELS


@MODELS.register_module()
class DCETransformer(nn.Module):
    def __init__(
        self,
        seq_len,
        use_grad=False,
        num_layers=2,
        embed_dims=32,
        num_heads=2,
        feedforward_channels=32,
        drop_rate=0
        ) -> None:

        super().__init__()
        self.use_grad = use_grad

        if self.use_grad:
            input_dim = 2
        else:
            input_dim = 1

        # position embeding for start and end time step
        self.register_buffer('pos_embed', torch.randn(1, seq_len+1, embed_dims))
        # self.register_buffer('pos_embed0', torch.randn(1, 1, embed_dims))
        # self.register_buffer('pos_embed1', torch.randn(1, 1, embed_dims))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.linear = nn.Linear(input_dim, embed_dims)
        self.layers = nn.ModuleList()

        # use ReLU to prevent negative outputs
        self.ktrans = nn.Sequential(nn.Linear(embed_dims, 1), nn.ReLU())
        self.kep = nn.Sequential(nn.Linear(embed_dims, 1), nn.ReLU())
        self.t0 = nn.Sequential(nn.Linear(embed_dims, 1), nn.ReLU())

        for _ in range(num_layers):
            self.layers.append(TransformerEncoderLayer(embed_dims=embed_dims, num_heads=num_heads, feedforward_channels=feedforward_channels))
    
    def forward(self, x, t=None):
        B = x.shape[0]

        if self.use_grad:
            # gradient of series
            grad = torch.gradient(x, dim=1)[0]
            x = torch.cat((x, grad), dim=2)

        x = self.linear(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)

        x = self.pos_embed + x
        x = self.drop_after_pos(x)
        
        for l in self.layers:
            x = l(x)
        x = x[:, -1, :]
        # x = x.mean(dim=1)
        
        ktrans = self.ktrans(x)
        kep = self.kep(x)
        t0 = self.t0(x)
        return ktrans, kep, t0


if __name__ == '__main__':
    import time
    model = MODELS.build(dict(type='DCETransformer', seq_len=75, use_grad=False)).cuda()
    x = torch.randn(160*160, 75, 2).cuda()
    tic = time.time()
    print(model(x))
    print((time.time() - tic) * 20)
