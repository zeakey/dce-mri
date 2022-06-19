dims = 16
model = dict(
    type='DCETransformer',
    seq_len=75,
    use_grad=False,
    num_layers=6,
    embed_dims=dims,
    num_heads=2,
    feedforward_channels=dims,
    drop_rate=0.3)


custom_imports = dict(imports=['models'], allow_failed_imports=False)
