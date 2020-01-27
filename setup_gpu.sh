#!/bin/bash

apt install wget python3-pip virtualenv

# create the python environment
virtualenv -p python3 gpu_env
source gpu_env/bin/activate
pip install -r requirements_gpu.txt

# Load the models
mkdir models
wget -P "./models/" -N "pollithy.com/rnn_model_fake_data.h5"
wget -P "./models/" -N "pollithy.com/DEM_model.h5"

mkdir experiments
# Load the data
DataSetsArray=(
	"Zylinder"
	"Pfeffer"
	"Kugeln"
	"Weizen"
	"DEM_Cylinder_115"
)

mkdir data
for datasetname in ${DataSetsArray[*]}; do
     echo $datasetname
     dataurl="pollithy.com/${datasetname}.zip"
     zipname="${datasetname}.zip"
     wget -P "./data/" -N "${dataurl}"
     unzip -n "./data/${zipname}" -d "./data/"
done