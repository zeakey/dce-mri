_base_ = ['../config.py']

aif = 'mixed'

model = dict(
    num_outputs=5,
    output_keys=['ktrans', 'kep', 't0', 'beta', 'noise_scale'])

sampler = dict(
    ktrans_sampler = dict(
        type='Beta',
        scale=1,
        concentration1=1.3,
        concentration0=13,
    ),
    kep_sampler = dict(
        type='Beta',
        scale=10,
        concentration1=1.3,
        concentration0=20,
    ),
    t0_sampler = dict(
        type='Uniform',
        scale=1,
        low=0,
        high=0.25,
    ),
    beta_sampler = dict(
        type='Uniform',
        low=0,
        high=1,
    ),
)

custom_imports = dict(imports=['models'], allow_failed_imports=False)
