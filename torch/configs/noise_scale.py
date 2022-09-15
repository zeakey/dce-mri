_base_ = ['./config.py']

model = dict(num_outputs=4)

custom_imports = dict(imports=['models'], allow_failed_imports=False)
