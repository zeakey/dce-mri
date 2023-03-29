_base_ = ['config.py']

aif = 'mixed'

model = dict(
    num_outputs=5,
    output_keys=['ktrans', 'kep', 't0', 'beta', 'noise_scale'])

sampler = dict(
    beta_sampler = dict(
        type='Uniform',
        low=0,
        high=1,
    ),
)

custom_imports = dict(imports=['models'], allow_failed_imports=False)
