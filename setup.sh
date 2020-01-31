#!/bin/bash

apt install wget python3-pip virtualenv openssh-server  # openssh-server for sftp



# ------------------------------------------
# Setup virtualenv and install requirements
# ------------------------------------------

read -p "GPU support? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  # GPU
  if [ ! -f "gpu_env" ]
  then
    virtualenv -p python3 "gpu_env"
  fi
  . gpu_env/bin/activate
  pip install -r "requirements_gpu.txt"
else
  # CPU
  if [ ! -f "cpu_env" ]
  then
    virtualenv -p python3 "cpu_env"
  fi
  . cpu_env/bin/activate
  pip install -r "requirements_cpu.txt"
fi

# ------------------------------------------
# Download data
# ------------------------------------------

if [ ! -f data ]
then
  # Create folders
  mkdir -p tmp_download_dir
  mkdir -p models
  mkdir -p data

  read -p "Do you want to download the data from ISAS i81server? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]
  then
      # ISAS
      read -p "Username for sftp on i81server: "  username
      echo "Username: $username"

      # Download bundle
      sftp "${username}@i81server:/mnt/data/user/home/inside-schuettgut/Datensaetze/proprak_ws1920.zip" tmp_download_dir/
  else
      # pollithy.com
      wget -P "./tmp_download_dir/" -N "pollithy.com/proprak_ws1920.zip"
  fi

  # Unzip and move the resources
  unzip tmp_download_dir/proprak_ws1920.zip -d tmp_download_dir/
  mv tmp_download_dir/proprak_ws1920/models/* ./models
  mv tmp_download_dir/proprak_ws1920/CsvDataSets/* ./data

  # Unzip the datasets
  ( cd data && unzip -n \*.zip )

  # remove tmp data
  rm -r tmp_download_dir
fi





