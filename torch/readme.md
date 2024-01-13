# DCE Analysis


## Environment setup
Python (>=3.09) is required to run this program.

[Miniconda](https://docs.conda.io/projects/miniconda) is recommended for package management.

If you have installed Miniconda (or anaconda), please follow these steps to setup your Python environment:

1. Run `conda create -n dce python=3.9`  to create an Python environment;
2. Run `conda activate dce` to activate the environment;
3. Run `pip3 install -r requirements.txt` to install all dependencies.

## How to use
After environment setup, you can run the example case:
`python3 simple-inference.py /path/to/your/dce/dicoms --save-path /path/to/where/you/want/to/save/results`.

The results will be saved to `/path/to/where/you/want/to/save/results`.
