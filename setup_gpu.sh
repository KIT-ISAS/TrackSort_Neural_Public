#!/bin/bash

apt install wget python3-pip virtualenv

# create the python environment
virtualenv -p python3 gpu_env
source gpu_env/bin/activate
pip install -r requirements_gpu.txt

bash download_data_and_models.sh

