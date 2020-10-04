#!/bin/bash
# ------------------------------------------
# Setup virtualenv and install requirements
# ------------------------------------------
read -p "Install virtualenv? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  apt install wget python3-pip virtualenv openssh-server  # openssh-server for sftp

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
      mkdir tmp_download_dir
      sftp "${username}@i81server.iar.kit.edu:/mnt/data/user/home/inside-schuettgut/Datensaetze/data_thumm_ss_20.zip" tmp_download_dir/

      # Unzip the data
      unzip tmp_download_dir/data_thumm_ss_20.zip -d ./

  fi
  # remove tmp data
  rm -r tmp_download_dir
fi
