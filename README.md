# NextStep-RNN

Using a RNN to predict next measurement of tracks.
Quick starts are provided in the "/Notebook" folder.

## Download data + checkpoints, install the libraries and run it with only two commands with a Linux shell on the CPU

`sh setup_cpu.sh`

 - defaults: `python main.py`
 - DEM: `python main.py --model_path "models/DEM_model.h5" --dataset_dir "data/DEM_cylinder.csv" --data_is_aligned False --is_loaded True  --rotate_columns True  --normalization_constant 1.0`


You can find the visualizations (step wise and as video) in the visualizations folder and you can set the hyperparams as described when typing "python main.py --help"

## Notebook

Follow these instructions to run notebooks on the gpu pc at the institute.
We use the docker images for [tensorflow](https://www.tensorflow.org/install/docker).

**Prerequisites**:
1. Install docker from source (not apt, because outdated)
2. Install latest nvidia drivers (no need to install CUDA, but be careful if your drivers are too old then docker is also not able to install the latest CUDA version. See the CUDA compatibility [table](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver))

**Install docker image**:
1. `docker pull tensorflow/tensorflow:latest-gpu-py3-jupyter`

**Install nvidia docker container toolkit**:
```
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```
(see [nvidia-docker](https://github.com/NVIDIA/nvidia-docker))

**Restart docker**:
`sudo systemctl restart docker`

**Run the docker container with jupyter**
```
# port 8888 for jupyter
# port 6006 for tensorboard
sudo docker run -p 8888:8888 -p 6006:6006 --gpus=all -it -v ~/Code:/tf --rm tensorflow/tensorflow:latest-gpu-py3-jupyter
```

Note: You have to mount your directory (in this example "~/Code" into "/tf" and this local Code directory
should contain this git repository. Then open the jupyter browser and navigate to the notebook.
