#!/bin/bash

apt install wget python3-pip virtualenv

# create the python environment
if [ -f cpu_env ]
then
	virtualenv -p python3 cpu_env
	. cpu_env/bin/activate
	pip install -r requirements_cpu.txt
fi

# Load the models
if [ ! -f models ]
then
	mkdir models
	wget -P "./models/" -N "pollithy.com/rnn_model_fake_data.h5"
	wget -P "./models/" -N "pollithy.com/DEM_model.h5"
fi

mkdir experiments
# Load the data
DataSetsArray=(
	"Zylinder"
	"Pfeffer"
	"Kugeln"
	"Weizen"
	"DEM_Cylinder_115"
)
if [ ! -f data ]
then
	mkdir data
	for datasetname in ${DataSetsArray[*]}; do
	     echo $datasetname
	     dataurl="pollithy.com/${datasetname}.zip"
	     zipname="${datasetname}.zip"
	     wget -P "./data/" -N "${dataurl}"
	     unzip -n "./data/${zipname}" -d "./data/"
	done
fi