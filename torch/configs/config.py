dims = 128
model = dict(
    type='DCETransformer',
    seq_len=75,
    use_grad=True,
    num_layers=3,
    embed_dims=dims,
    num_heads=4,
    feedforward_channels=dims*3,
    drop_rate=0.3)

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.pos_embed': dict(decay_mult=0.0),
    })
optimizer = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=5e-5,
    paramwise_cfg=paramwise_cfg)

sampler = dict(
    ktrans_sampler = dict(
        type='Beta',
        scale=1,
        concentration1=1.3,
        concentration0=13,
    ),
    kep_sampler = dict(
        type='Beta',
        scale=1,
        concentration1=1.3,
        concentration0=20,
    ),
    t0_sampler = dict(
        type='Beta',
        scale=1,
        concentration1=1,
        concentration0=1,
    ),
)

loss = 'l1'

aif = 'parker'

custom_imports = dict(imports=['models'], allow_failed_imports=False)
