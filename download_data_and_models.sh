#!/bin/bash


read -p "Do you want to download the data from ISAS i81server? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # ISAS

    read -p "Username for sftp on i81server: "  username
    echo "Username: $username"

    # Create folders
    mkdir -p tmp_download_dir
    mkdir -p models
    mkdir -p data

    # Download bundle
    sftp "${username}@i81server:/mnt/data/user/home/inside-schuettgut/Datensaetze/proprak_ws1920.zip" tmp_download_dir/
    unzip tmp_download_dir/proprak_ws1920.zip -d tmp_download_dir/
    mv tmp_download_dir/proprak_ws1920/models/* ./models
    mv tmp_download_dir/proprak_ws1920/CsvDataSets/* ./data

    # Unzip the datasets
    ( cd data && unzip -n \*.zip )

    # remove tmp data
    rm -r tmp_download_dir

else
    # pollithy.com

    # Load the models
    mkdir -p models
    wget -P "./models/" -N "pollithy.com/rnn_model_fake_data.h5"
    wget -P "./models/" -N "pollithy.com/DEM_model.h5"

    # Load the data
    DataSetsArray=(
      "Zylinder"
      "Pfeffer"
      "Kugeln"
      "Weizen"
      "DEM_Cylinder_115"
    )

    mkdir -p data

    for datasetname in ${DataSetsArray[*]}; do
         echo "$datasetname"
         dataurl="pollithy.com/${datasetname}.zip"
         zipname="${datasetname}.zip"
         wget -P "./data/" -N "${dataurl}"
         unzip -n "./data/${zipname}" -d "./data/"
    done
fi




