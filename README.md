# TrackSort Neural
Implementation of multitarget tracking and separation prediction for optical belt sorters.
**Key features:**
* Multiple experts for tracking and separation prediction
  * Kalman filters (KF) - constant velocity (CV) and constant acceleration (CA)
  * Multilayer Perceptrons (MLP)
  * Long-short term memory networks (LSTM)
* Expert combination methods to combine the predictions of multiple experts
  * Simple ensemble
  * Covariance weighting
  * SMAPE weighting
  * **Mixture of experts (ME)**
* Uncertainty prediction of the separation predictions
* Uncertainty evaluation and calibration with **new SENCE method**
* Large variety of evaluation functions for tracking and separation prediction

**Outline:**
* [Motivation](#motivation)
* [Code Structure](#code-structure)
* [Installation](#installation)
* [Run Command Options](#run-command-options)
* [Common Run Commands](#common-run-commands)
  * [Single-target Tracking with KF Models](#single-target-tracking-with-kf-models)
  * [Single-target Tracking with MLP Model](#single-target-tracking-with-mlp-model)
  * [Single-target Tracking with RNN Model](#single-target-tracking-with-rnn-model)
  * [Single-target Tracking with ME Model](#single-target-tracking-with-me-model)
  * [Multitarget Tracking with KF Model on DEM Cylinder Simulation Data without Additional Noise](#multitarget-tracking-with-kf-model-on-dem-cylinder-simulation-data-without-additional-noise)
  * [Separation Prediction with Motion Models](#separation-prediction-with-motion-models)
  * [Separation Prediction with MLP Model](#separation-prediction-with-mlp-model)
  * [Separation Prediction with RNN Model](#separation-prediction-with-rnn-model)
  * [Separation Prediction with ME Model](#separation-prediction-with-me-model)
  * [Separation Prediction with MLP Model and Uncertainty Prediction](#separation-prediction-with-mlp-model-and-uncertainty-prediction)
* [Config files](#config-files)
* [Contribute](#contribute)

## Motivation
This framework allows us to test and compare a vast variety of experts for the tasks of tracking and separation prediction in optical belt sorters.
The code includes every step from the data pipeline to the training and testing of the experts.
With this code we are able to train a state of the art ME approach that is able to outperform single experts in every sorting scenario and improves the sorting accuracy in dynamic scenarios significantly.
The implemented uncertainty prediction for the separation prediction not only improves the accuracy of the neural networks but is also a potential key input for a dynamic nozzle control.

## Code Structure
The idea of the expert combination in tracking can be explained with the following figure.
![ME structure](/images/ME_Structure_Tracking.png)
For every new measurement of a particle, the experts, here one KF, one LSTM, and one MLP, predict the next particle position.
The gating network then assigns weights to each expert prediction. The combined expert prediction is the weighted sum of expert predictions.
The combined prediction is used in the data association to match a prediction to a new measurement and the cycle restarts.

An example work flow with the most important classes and functions is displayed in the following figure. The sequence diagram shows the process of training multiple experts in parallel and afterwards training a gating network using the predictions of the trained experts. (Note: Often, it is not recommended to train experts in parallel because they then use the same number of training epochs. It is more reliable to train each expert separately and then load them in.)
![Sequence diagramm](/images/sequence_diagramm.svg)
The model manager has an expert manager object and a gating network object. The expert manager manages all experts for training and testing. The gating network assigns weights to the experts. The model manager is responsible for the interaction between the experts and the gating network. All training and testing calls start from the model manager and are then sent to either the expert manager or gating network.

## Installation

### Setup Git Repo
1. `git clone https://github.com/KIT-ISAS/TrackSort_Neural.git` to clone this repo to the folder of your choice.
2. `cd TrackSort_Neural` to work in the newly cloned project.
2. `git checkout jakob_master` to checkout this branch.
3. `. setup.sh` to automatically install virtualenv and download the data and models from the ISAS server 
    --> Access needed! Please contact marcel.reith-braun@kit.edu for external access.
    `Install virtualenv? (y/n)`, this is not necessarily needed if you are planning to work with GPU docker.
    `Do you want to download the data from ISAS i81server? (y/n)` --> `y`
    Enter username and password for the ISAS server access.
    Now the files will be downloaded and unziped.
    
### Run with CPU
`python main.py` should work when you select `y` at the question `Install virtualenv? (y/n)` in `setup.sh`.

### Run with GPU and Docker (recommended)
1. `docker build -t py3-tensorflow/jakob-thumm:v1 . ` builds the docker image
2. `docker run --gpus '"device=0"' -it -v "/home/thumm/masterthesis/code/TrackSort_Neural:/home/TrackSort_Neural" py3-tensorflow/jakob-thumm:v1` build a interactive container
3. `cd home/TrackSort_Neural/` cd into the working folder in the running container
4. `python main.py` to run the programm

### Run Time-Intensive Training Scripts on the GPU-Server
1. `ssh [username]@i81-gpu-server.iar.kit.edu` replace `[username]` with your username.
2. setup the git repo as described in [Setup Git Repo](#setup-git-repo)
3. `docker build -t py3-tensorflow/jakob-thumm:v1 . ` builds the docker image
3. `tmux` to start a tmux session (Does not quit when closing the terminal if properly detached.)
4. `docker run --gpus '"device=0"' -d -v "/home/TrackSort_Neural:/home/TrackSort_Neural" py3-tensorflow/jakob-thumm:v1` to start a docker container in detached state. Change `"/home/TrackSort_Neural:` to whereever you put your project folder on the server.
5. `docker container ls` to show the currently running docker containers --> Copy the id of `py3-tensorflow/jakob-thumm:v1`
6. `docker exec -d [docker_id] /bin/sh -c "cd home/TrackSort_Neural/scripts; ./run_train_tracking_DEM_cylinder.sh"` to run a bash script in the container. Replace `[docker_id]` with the copied docker id.
7. `tmux detach` detach the tmux session. Now you can close the terminal and let the GPU server do its magic.

## Run Command Options
* `-h, --help`  -  show this help message and exit
* `--config_path` - Path to config file including information about experts, gating network and weighting function.
* `--is_loaded` - Whether the models should be loaded or trained.
* `--is_loaded_gating_network` - Whether the gating network should be loaded or trained.
* `--dataset_dir` - The directory of the data set. Only needed for CsvDataset.
* `--dataset_type` {FakeDataset,CsvDataset} - The type of the dataset.
* `--result_path` - The path where the results are stored.
* `--batch_size` - The batchsize, that is used for training and inference
* `--evaluation_ratio` - The ratio of data used for evaluation.
* `--test_ratio` - The ratio of data used for the final unbiased test.
* `--num_timesteps` - The number of timesteps of the dataset. Necessary for FakeDataset.
* `--num_train_epochs` - Number of epochs in expert training.
* `--improvement_break_condition` - Break training if test loss on every expert does not improve by more than this value.
* `--nan_value` - The Nan value, that is used by the DataManager.
* `--matching_algorithm` {local,global} - The algorithm, that is used for matching in the MTT.
* `--distance_threshold` - The threshold used for the matching with the artificial measurements and predictions.
* `--birth_rate_mean` - The birth rate mean value, that is used by the DataManager.
* `--birth_rate_std` - The birth rate std value, that is used by the DataManager.
* `--normalization_constant` - Normalization value (Leave this out for an automatic calculation).
* `--evaluate_every_n_epochs` - Show training metric every n training epochs.
* `--time_normalization_constant` - Normalization factor for temporal prediction.
* `--mlp_input_dim` - The dimension of input points for the MLP.
* `--separation_mlp_input_dim` - The dimension of input points for the separation MLP.
* `--data_is_aligned` - Whether the data used by the DataManger is aligned or not (Use False for DEM simulation and True for real-world data).
* `--rotate_columns` - Set this to True if the order of columns in your csv is (x, y). Default is (y, x) (Use True for DEM simulation and False for real-world data).
* `--n_folded_cross_evaluation` - Change this from -1 to i.e. 5 to activate cross evaluation (Should be activated when analyzing uncertainty prediction!).
* `--tracking` - Perform tracking.
* `--separation_prediction` - Perform separation predcition.
* `--uncertainty_prediction` - Perform uncertainty predcition (Only possible for separation prediction right now).
* `--verbosity` {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}
* `--logfile` - Path to logfile. Leave this out to only log to console. This is recommended when training on the GPU server.
* `--additive_noise_stddev` - Standard deviation of additional Gaussian white noise. This can be used to add noise to the otherwise perfect simulation data.
* `--virtual_belt_edge_x_position` - Position of the virtual belt end.
* `--virtual_nozzle_array_x_position` - Position of the virtual nozzle array.
* `--min_measurements_count` - Ignore tracks with less measurements.
* `--execute_evaluation` - Run evaluation after training/loading or not.
* `--execute_multi_target_tracking` - Run multi-target tracking after training/loading or not.
* `--evaluate_mlp_mask` - Masks every model with a mlp masks to compare MLPs with other models in testing function.
* `--no_show` - Set this to True if you do not want to show evaluation graphics and only save them.
* `--visualize_multi_target_tracking` - You can generate nice videos of the multitarget tracking.

## Common Run Commands
### Single-target Tracking with KF Models
```
./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --tracking True \
  --uncertainty_prediction False \
  --config_path configs/pepper/test_kf_pepper.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --num_train_epochs 0 \
  --batch_size 64 \
  --result_path results/pepper_all/kf_tracking/ \
  --execute_evaluation True \
  --evaluate_mlp_mask False
```
### Single-target Tracking with MLP Model
```
./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --tracking True \
  --uncertainty_prediction False \
  --config_path configs/pepper/train_pepper_mlp.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --num_train_epochs 2000 \
  --batch_size 64 \
  --evaluate_every_n_epochs 20 \
  --result_path results/pepper_all/mlp_tracking/ \
  --execute_evaluation True \
  --evaluate_mlp_mask False
```
### Single-target Tracking with RNN Model
```
./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --tracking True \
  --uncertainty_prediction False \
  --config_path configs/pepper/train_pepper_rnn.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --num_train_epochs 650 \
  --batch_size 64 \
  --evaluate_every_n_epochs 20 \
  --result_path results/pepper_all/rnn_tracking/ \
  --execute_evaluation True \
  --evaluate_mlp_mask False
```
### Single-target Tracking with ME Model
```
./main.py \
  --is_loaded True \
  --is_loaded_gating_network False \
  --tracking True \
  --uncertainty_prediction False \
  --config_path configs/pepper/pepper_me_pos_id_weighting.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --num_train_epochs 650 \
  --batch_size 64 \
  --evaluate_every_n_epochs 20 \
  --result_path results/pepper_all/mixture_of_experts_pos_id_tracking/ \
  --execute_evaluation True \
  --evaluate_mlp_mask False
```
### Multitarget Tracking with KF Model on DEM Cylinder Simulation Data without Additional Noise
```
./main.py \
  --is_loaded True \
  --is_loaded_gating_network True \
  --tracking True \
  --uncertainty_prediction False \
  --config_path configs/DEM_cylinder/train_cylinder_kf_cv_no_noise.json \
  --dataset_dir 'data/DEM_Data/csv_converted_FOV/Cylinders_115_200Hz.csv' \
  --data_is_aligned False \
  --rotate_columns True \
  --normalization_constant 1.0 \
  --additive_noise_stddev 0 \
  --num_train_epochs 0 \
  --batch_size 128 \
  --evaluate_every_n_epochs 20 \
  --result_path results/DEM_cylinder/kf_cv_no_noise/ \
  --execute_evaluation True \
  --evaluate_mlp_mask False \
  --execute_multi_target_tracking True
```
### Separation Prediction with Motion Models
```
./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --tracking False \
  --separation_prediction True \
  --uncertainty_prediction False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --config_path configs/pepper/test_pepper_sep_kf.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1 \
  --batch_size 128 \
  --result_path results/pepper_all/kf_separation/ \
  --execute_evaluation True \
  --execute_multi_target_tracking False
```
### Separation Prediction with MLP Model
```
./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --tracking False \
  --separation_prediction True \
  --uncertainty_prediction False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --config_path configs/pepper/train_pepper_mlp_sep.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 2000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 20 \
  --result_path results/pepper_all/mlp_separation/ \
  --execute_evaluation True \
  --execute_multi_target_tracking False
```
### Separation Prediction with RNN Model
```
./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --tracking False \
  --separation_prediction True \
  --uncertainty_prediction False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --config_path configs/pepper/train_pepper_rnn_hyb.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 20 \
  --result_path results/pepper_all/rnn_separation/ \
  --execute_evaluation True \
  --execute_multi_target_tracking False
```
### Separation Prediction with ME Model
```
./main.py \
  --is_loaded True \
  --is_loaded_gating_network False \
  --tracking False \
  --separation_prediction True \
  --uncertainty_prediction False \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --config_path configs/pepper/train_pepper_sep_gating_me.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --time_normalization_constant 15.71 \
  --num_train_epochs 1000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 20 \
  --result_path results/pepper_all/ME_separation/ \
  --execute_evaluation True \
  --execute_multi_target_tracking False
```
### Separation Prediction with MLP Model and Uncertainty Prediction
```
./main.py \
  --is_loaded False \
  --is_loaded_gating_network True \
  --tracking False \
  --separation_prediction True \
  --uncertainty_prediction True \
  --virtual_belt_edge_x_position 800 \
  --virtual_nozzle_array_x_position 1550 \
  --config_path configs/pepper/train_pepper_mlp_sep_uncertainty.json \
  --dataset_dir 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv' \
  --n_folded_cross_evaluation 5 \
  --time_normalization_constant 15.71 \
  --separation_mlp_input_dim 7 \
  --num_train_epochs 4000 \
  --batch_size 128 \
  --evaluate_every_n_epochs 20 \
  --result_path results/pepper_all/mlp_separation_uncertainty/ \
  --execute_evaluation True \
  --execute_multi_target_tracking False
```

## Config Files
Json file options:
* `batch_size` - Batch size of experts (Especially neccessary for RNN)
* `experts` - The experts for tracking or separation prediction
  * `expert name` - The name of the expert (e.g. "MLP")
    * `type` - "KF", "MLP", or "RNN"
    * `sub_type` - Only needed for KF, "CV" or "CA"
    * `model_path` - Path to load or save the model
    * `is_separator` - Separation prediction or tracking expert
    * `model_options` - Options for the model
    * `state_options` - Only for KF, default state options
* `gating` - The gating networkork
  * `type` - "Simple_Ensemble", "Covariance_Weighting", "SMAPE_Weighting", or "Mixture_of_Experts"
  * `model_path` - Path to save or load gating network
  * `options` - Options for training (especially for ME)
* `gating_separation` - The gating networkork for the separation prediction
  * `type` - "Simple_Ensemble", "Covariance_Weighting", "SMAPE_Weighting", or "Mixture_of_Experts"
  * `model_path` - Path to save or load gating network
  * `options` - Options for training (especially for ME)

## Contribute
If you are intereseted in contributing to this project, please contact 
* Marcel Reith-Braun https://isas.iar.kit.edu/de/Mitarbeiter_Reith-Braun.php
* Florian Pfaff https://isas.iar.kit.edu/de/Mitarbeiter_Pfaff.php
* Jakob Thumm jakob.thumm@web.de

We are happy, that you decided to work on this project. Please keep the code clean, so:
* Make sure to comment every function in docstring. 
* Comment your code so someone can work with it after you left.
* Do not push uncommented code.
* Delete unneccessary code when you are done with it.

Thank you and have fun.
