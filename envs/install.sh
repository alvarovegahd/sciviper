# Run line by line
conda env create -f envs/vipergpt_greatlakes.yml # for trace, temporarily change the name to vipergpt_trace
conda activate vipergpt_trace
pip install -r envs/requirements_greatlakes.txt

# If you need to upgrade gcc
conda install -c conda-forge compilers

# Reference:
#   https://anaconda.org/nvidia/cuda-nvcc/labels
#   https://anaconda.org/nvidia/cuda-cudart-dev/labels
conda install -c nvidia cuda-nvcc=12.1.66
conda install -c nvidia cuda-cudart-dev=12.1.55

# And run the following, INSIDE A SRUN COMMAND WITH GPU
export CUDA_HOME="$CONDA_PREFIX"
cd GLIP
python setup.py clean --all build develop --user
cd -

# For trace
cd src/Trace
pip install -e ."[autogen]"
pip install datasets ray ipdb

# # Or alternatively:
# pip install trace-opt

# Since Trace requires pydantic > 1.10 and <3.0, and ViperGPT uses pydantic==1.10, we need to modify the following line in the source file.
# Replace the following line in the inflect package
code $CONDA_PREFIX/lib/python3.10/site-packages/inflect/__init__.py
# try:
#     from pydantic.typing import Annotated
# except ImportError:
#     from typing import Annotated
