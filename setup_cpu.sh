#!/bin/bash

apt install wget python3-pip virtualenv

# create the python environment
if [ ! -f cpu_env ]
then
	virtualenv -p python3 cpu_env
	. cpu_env/bin/activate
	pip install -r requirements_cpu.txt
fi

bash download_data_and_models.sh