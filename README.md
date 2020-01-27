# NextStep-RNN

**Content:**

- Using a RNN to predict next measurement of tracks.
- Using a RNN to make separation predictions
- Python implementation of data assocation (local and global nearest neighbour)
Quick starts are provided in the "/Notebook" folder.

## Run with CPU

1. `sh setup_cpu.sh`
2. `python main.py`

You can find the visualizations (step wise and as video) in the visualizations folder and you can set the hyperparams as described when typing "python main.py --help"

## Run with GPU (Docker)

1. Pull docker image `docker pull tensorflow/tensorflow:2.1.0-gpu-py3-jupyter`
2. Clone repo: `git clone https://github.com/sidney1505/next_step_rnn ~/next_step_rnn`
3. Run docker with mounted home dir: `docker run -it -v $PWD:/tf --gpus "device=0" tensorflow/tensorflow:2.1.0-gpu-py3`
4. Inside the container: `cd next_step_rnn && bash setup_gpu.sh`
5. Inside the container: `source gpu_env/bin/activate`
6. Inside the container: `python main.py ...`

## Notebooks:  Run with GPU (Docker)

Follow these instructions to run notebooks on the gpu pc at the institute.

1. `docker pull tensorflow/tensorflow:2.1.0-gpu-py3-jupyter`
2. Clone repo: `git clone https://github.com/sidney1505/next_step_rnn ~/next_step_rnn`
3. `docker run -it -p 8888:8888 -v $PWD:/tf --gpus "device=0" tensorflow/tensorflow:2.1.0-gpu-py3-jupyter`
4. (ssh tunneling on remote machine: `ssh -L 8888:127.0.0.1:8888 proprak7@i81-gpu-server`)

## python main.py ...

This is the main starting point and it has a lot of settings.
See: `python main.py --help`

### Quickstarts for main.py

 - DEM Zylinder: `python main.py --model_path "models/DEM_model.h5" --dataset_dir "data/DEM_cylinder.csv" --data_is_aligned False --is_loaded True  --rotate_columns True  --normalization_constant 1.0`
 - CSV Pfeffer ...