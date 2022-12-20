_base_ = ['./config.py']

model = dict(
    num_outputs=4,
    output_keys=['ktrans', 'kep', 't0', 'noise_scale'])

custom_imports = dict(imports=['models'], allow_failed_imports=False)
