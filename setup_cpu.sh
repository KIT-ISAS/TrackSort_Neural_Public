#!/bin/bash

virtualenv -p python3 cpu_env
. cpu_env/bin/activate
pip install -r requirements_cpu.txt


# Load models
mkdir models
wget -P "./models/" -N "pollithy.com/rnn_model_fake_data.h5"


# Load data
mkdir data

# wrong files
# wget -P "./data/" -N "pollithy.com/DEM_Zylinder.csv"
# wget -P "./data/" -N "pollithy.com/DEM_holzkugeln.csv"

DataSetsArray=("Zylinder"  "Pfeffer"  "Kugeln"  "Weizen")

for datasetname in ${DataSetsArray[*]}; do
     echo $datasetname
     dataurl="pollithy.com/${datasetname}.zip"
     zipname="${datasetname}.zip"
     wget -P "./data/" -N "${dataurl}"
     unzip -n "./data/${zipname}" -d "./data/"
done
