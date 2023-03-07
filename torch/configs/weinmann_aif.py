_base_ = ['./config.py']

model = dict(num_outputs=3,)

sampler = dict(
    t0_sampler = dict(
        type='Uniform',
        scale=1,
        low=0,
        high=1,
    ),
)
aif = 'weinmann'

custom_imports = dict(imports=['models'], allow_failed_imports=False)
