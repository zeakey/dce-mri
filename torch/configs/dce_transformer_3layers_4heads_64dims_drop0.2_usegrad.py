dims = 64
model = dict(
    type='DCETransformer',
    seq_len=75,
    use_grad=True,
    num_layers=3,
    embed_dims=dims,
    num_heads=4,
    feedforward_channels=dims*3,
    drop_rate=0.2)

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.pos_embed': dict(decay_mult=0.0),
    })

# optimizer = dict(
#     type='AdamW',
#     lr=1e-4,
#     weight_decay=0.05,
#     eps=1e-8,
#     betas=(0.9, 0.999),
#     paramwise_cfg=paramwise_cfg)

optimizer = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=5e-5,
    paramwise_cfg=paramwise_cfg)

custom_imports = dict(imports=['models'], allow_failed_imports=False)
