import code  # code.interact(local=dict(globals(), **locals()))
import os
import shutil
import argparse
import json
import datetime
import logging

import tensorflow as tf
import numpy as np

#from moviepy.editor import ImageSequenceClip
from track_manager import TrackManager
from model_manager import ModelManager
from data_association import DataAssociation
from data_manager import FakeDataSet, CsvDataSet
from evaluator import Evaluator
# Test
from cv_model import *

tf.get_logger().setLevel('ERROR')
#tf.enable_eager_execution()
parser = argparse.ArgumentParser()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# the possible arguments you can give to the model
parser.add_argument('--is_loaded', type=str2bool, default=True,
                    help='Whether the model is loaded or created + trained.')
parser.add_argument('--model_path', default='models/rnn_model_fake_data.h5',
                    help='The path where the model is stored or loaded from.')
parser.add_argument('--matching_algorithm', default='global', choices=['local', 'global'],
                    help='The algorithm, that is used for matching.')
parser.add_argument('--dataset_dir', default='data/Pfeffer/trackSortResultPfeffer/001_trackHistory_NothingDeleted.csv',
                    help='The directory of the data set. Only needed for CsvDataset.')
parser.add_argument('--dataset_type', default='CsvDataset', choices=['FakeDataset', 'CsvDataset'],
                    help='The type of the dataset.')
parser.add_argument('--distance_threshold', type=float, default=0.02,
                    help='The threshold used for the matching with the artificial measurements and predictions')
parser.add_argument('--config_path', default="configs/default_config.json",
                    help='Path to config file including information about experts, gating network and weighting function.')
parser.add_argument('--batch_size', type=int, default=64, help='The batchsize, that is used for training and inference')
parser.add_argument('--num_timesteps', type=int, default=350,
                    help='The number of timesteps of the dataset. Necessary for FakeDataset.')
parser.add_argument('--num_train_epochs', type=int, default=100, help='Only necessary, when model is trained.')
parser.add_argument('--lr_decay_after_epochs', type=int, default=150, help='When to decrease the lr by lr_decay_factor')
parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='When learning rate should be decreased, '
                                                                       'multiply with this')
parser.add_argument('--nan_value', type=float, default=0.0, help='The Nan value, that is used by the DataManager')
parser.add_argument('--birth_rate_mean', type=float, default=5.0,
                    help='The birth_rate_mean value, that is used by the DataManager')
parser.add_argument('--birth_rate_std', type=float, default=2.0,
                    help='The birth_rate_std value, that is used by the DataManager')
parser.add_argument('--normalization_constant', type=float, default=None, help='Normalization value')
parser.add_argument('--evaluate_every_n_epochs', type=int, default=20)
parser.add_argument('--time_normalization_constant', type=float, default=22.0, help='Normalization for time prediction')
parser.add_argument('--min_number_detections', type=int, default=6,
                    help='The min_number_detections value, that is used by the DataManager')
parser.add_argument('--input_dim', type=int, default=2, help='The input_dim value, that is used by the DataManager')
parser.add_argument('--data_is_aligned', type=str2bool, default=True,
                    help='Whether the data used by the DataManger is aligned or not.')
parser.add_argument('--rotate_columns', type=str2bool, default=False,
                    help='Set this to true if the order of columns in your csv is (x, y). Default is (y, x)')
parser.add_argument('--run_hyperparameter_search', type=str2bool, default=False,
                    help='Whether to run the hyperparameter search or not')
parser.add_argument('--test_noise_robustness', type=str2bool, default=False,
                    help='Should the Dataset be tested with multiple noise values?')
parser.add_argument('--separation_prediction', type=str2bool, default=False,
                    help='Should the RNN also predict the separation?')
parser.add_argument('--verbosity', default='INFO', choices=logging._nameToLevel.keys())

parser.add_argument('--additive_noise_stddev', type=float, default=0.0)

parser.add_argument('--virtual_belt_edge_x_position', type=float, default=800,
                    help='Where does the virtual belt end?')
parser.add_argument('--virtual_nozzle_array_x_position', type=float, default=1550,
                    help='Where should the virtual nozzle array be?')
parser.add_argument('--num_units_first_rnn', type=int, default=1024,
                    help='Where should the virtual nozzle array be?')
parser.add_argument('--num_units_second_rnn', type=int, default=16,
                    help='Where should the virtual nozzle array be?')
parser.add_argument('--clear_state', type=str2bool, default=True,
                    help='Whether a new track should be initialized with empty state?')
parser.add_argument('--overwriting_activated', type=str2bool, default=True,
                    help='Whether batches of the RNN are reused')

args = parser.parse_args()

global_config = {
    'separation_prediction': args.separation_prediction,
    'clear_state': args.clear_state,
    'time_normalization_constant': args.time_normalization_constant,
    'virtual_belt_edge_x_position': args.virtual_belt_edge_x_position,
    'virtual_nozzle_array_x_position': args.virtual_nozzle_array_x_position,
    'only_last_timestep_additional_loss': True,

    'is_loaded': args.is_loaded,
    'model_path': args.model_path,
    'distance_threshold': args.distance_threshold,
    'config_path': args.config_path,
    'batch_size': args.batch_size,
    'matching_algorithm': args.matching_algorithm,
    #
    'Track': {
        'initial_is_alive_probability': 0.5,
        'is_alive_decrease': 0.25,
        'is_alive_increase': 0.5,
    },
    #
    'num_timesteps': args.num_timesteps,
    'dataset_type': args.dataset_type,
    #
    'CsvDataSet': {
        'glob_file_pattern': args.dataset_dir,
        'min_number_detections': args.min_number_detections,
        'nan_value': args.nan_value,
        'input_dim': args.input_dim,
        'data_is_aligned': args.data_is_aligned,
        'birth_rate_mean': args.birth_rate_mean,
        'birth_rate_std': args.birth_rate_std,
        'rotate_columns': args.rotate_columns,
        'normalization_constant': args.normalization_constant,
        'additive_noise_stddev': args.additive_noise_stddev
    },
    'num_train_epochs': args.num_train_epochs,
    'evaluate_every_n_epochs': args.evaluate_every_n_epochs,
    'lr_decay_after_epochs': args.lr_decay_after_epochs,
    'lr_decay_factor': args.lr_decay_factor,

    'state_overwriting_started': False,
    'overwriting_activated': args.overwriting_activated,
    'verbose': 1,
    'visualize': False,
    'run_hyperparameter_search': args.run_hyperparameter_search,
    'debug': False,
    'test_noise_robustness': args.test_noise_robustness,
    'experiment_series': 'independent',
    'is_alive_probability_weighting': 0.0,
    'positional_probabilities': 0.0
}

# setup logging
log_level = int(logging._nameToLevel[args.verbosity])
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)
logging.log(log_level, "LOG LEVEL: %s", log_level)


def run_global_config(global_config, experiment_series_names=''):
    # Create paths for run
    now = datetime.datetime.now()
    experiment_name = now.strftime("%Y_%m_%d__%H_%M_%S")
    global_config['experiment_name'] = experiment_name

    # create directories
    global_config['results_path'] = os.path.join("results", experiment_series_names, experiment_name)
    global_config['experiment_path'] = os.path.join(global_config['results_path'], 'experiments')
    global_config['diagrams_path'] = os.path.join(global_config['results_path'], 'visualizations', 'diagrams')
    global_config['visualization_path'] = os.path.join(global_config['results_path'], 'visualizations',
                                                       'matching_visualization')

    for config_key in ['results_path', 'experiment_path', 'diagrams_path', 'visualization_path']:
        os.makedirs(global_config[config_key], exist_ok=True)

    # file paths
    global_config['visualization_video_path'] = os.path.join(global_config['visualization_path'],
                                                             'matching_visualization_vid.mp4')
    global_config['json_file'] = os.path.join(global_config['experiment_path'], 'config.json')

    ## Import data
    if global_config['dataset_type'] == 'FakeDataset':
        data_source = FakeDataSet(timesteps=global_config['num_timesteps'], batch_size=global_config['batch_size'])
    elif global_config['dataset_type'] == 'CsvDataset':
        data_source = CsvDataSet(**global_config['CsvDataSet'])
        global_config['num_timesteps'] = data_source.get_num_timesteps()
    
    # calc mean vel in first time step

    
    sum_v = 0
    c = 0
    for track in data_source.aligned_track_data:
        c = c+1
        sum_v = sum_v + (track[1,0] - track[0,0])
    dt = 0.005
    mean_v = (sum_v/c)/dt
    var_sum = 0
    for track in data_source.aligned_track_data:
        var_sum = var_sum + ((track[1,0] - track[0,0])/dt - mean_v)**2
    var_v = var_sum/c
    """
    cv_model = CV_Model(dt = dt, s_w = 10E7)
    # Test CV model
    c = 0
    error = 0
    for track in data_source.aligned_track_data:
        cv_state = CV_State(track[0], velocity_guess=15000, velo_var_x=10E6, velo_var_y = 10E8)
        cv_model.update(cv_state)
        cv_model.predict(cv_state)
        for i in range(1,track.shape[0]):
            pos = track[i,:]
            if np.array_equal(pos,np.array([0, 0])) == False:
                state_pos = [cv_state.get_pos().item(0), cv_state.get_pos().item(1)]
                error = error + np.sqrt((pos[0]-state_pos[0]) ** 2 + (pos[1]-state_pos[1]) ** 2)
                # update
                cv_model.update(cv_state, pos)
                # predict
                cv_model.predict(cv_state)
                c = c+1

    print(error/c)
    """

    ## Import model config to json tree
    # TODO: Create json schema to check config validity
    with open(global_config["config_path"]) as f:
        model_config = json.load(f)
        
    # [TEST] of training
    #global_config["is_loaded"] = False
    ## Initialize models
    model_manager = ModelManager(global_config, data_source, model_config, global_config["is_loaded"], data_source.longest_track)

    ## Train or import model
    if global_config["is_loaded"]:
        model_manager.load_models(global_config["model_path"])
    else:
        # Seperate dataset into train and test set
        # TODO: Fix random seed 
        dataset_train, dataset_test = data_source.get_tf_data_sets_seq2seq_data(normalized=True)
        # Train models
        model_manager.train_models(dataset_train, global_config["num_train_epochs"])
        # Test models


    ## Init tracks
    track_manager = TrackManager(global_config)

    data_association = DataAssociation(global_config)
    particles = data_source.get_particles()
    tracks = data_association.associate_data(data_source, track_manager, model_manager)

    if global_config['visualize']:
        shutil.rmtree(global_config['visualization_video_path'], ignore_errors=True)
        clip = ImageSequenceClip(global_config['visualization_path'], fps=4)
        clip.write_videofile(global_config['visualization_video_path'], fps=4)

    evaluator = Evaluator(global_config, particles, tracks)
    accuracy_of_the_first_kind = 1.0 - evaluator.error_of_first_kind()
    accuracy_of_the_second_kind = 1.0 - evaluator.error_of_second_kind()
    score = 2 * accuracy_of_the_first_kind * accuracy_of_the_second_kind / (
            accuracy_of_the_first_kind + accuracy_of_the_second_kind)

    # save the current config
    global_config['score'] = score
    global_config['accuracy_of_the_first_kind'] = accuracy_of_the_first_kind
    global_config['accuracy_of_the_second_kind'] = accuracy_of_the_second_kind

    with open(global_config['json_file'], 'w') as file_:
        json.dump(global_config, file_, sort_keys=True, indent=4)

    del data_association
    del particles
    del tracks
    del global_config

    return score, accuracy_of_the_first_kind, accuracy_of_the_second_kind


if not global_config['run_hyperparameter_search']:
    if not global_config['test_noise_robustness']:
        score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(global_config)
        logging.info('data association finished!')
        code.interact(local=dict(globals(), **locals()))
    else:
        logging.info('test robustness against noise!')
        now = datetime.datetime.now()
        experiment_series = 'noise_robustness_' + now.strftime("%Y_%m_%d__%H_%M_%S")
        global_config['experiment_series'] = experiment_series
        result_list = []
        for noise in [0.0, 0.0003, 0.0005, 0.0008, 0.001]:
            worked = False
            while not worked:
                try:
                    tf.keras.backend.clear_session()
                    current_config = global_config.copy()
                    current_config['CsvDataSet']['additive_noise_stddev'] = noise
                    score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(
                        current_config, experiment_series_names=experiment_series)
                    result_list.append([noise, score, accuracy_of_the_first_kind, accuracy_of_the_second_kind])
                    worked = True
                except ValueError as value_error:
                    logging.error("duplicate key values in evaluation")
                    logging.error(str(value_error))

        logging.debug(str(result_list))

        A = np.array(result_list)
        np.savetxt(os.path.join("results", experiment_series, "noise_robustness.csv"), A)

        logging.info('robustness test finished!')
        code.interact(local=dict(globals(), **locals()))

else:
    now = datetime.datetime.now()
    global_config['experiment_series'] = 'hyperparamsearch_' + now.strftime("%Y_%m_%d__%H_%M_%S")
    os.makedirs('experiments/' + global_config['experiment_series'])
    dt = global_config['distance_threshold']
    best_score = 0.0
    distance_threshold_candidates = [0.25 * dt, 0.5 * dt, dt, 2.0 * dt, 4.0 * dt]
    candidate_scores = []
    best_candidate = distance_threshold_candidates[0]
    for candidate in distance_threshold_candidates:
        logging.debug('run distance_threshhold %s', str(candidate))
        global_config['distance_threshold'] = candidate
        current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(global_config, \
            experiment_series_names=global_config['experiment_series'])
        #
        candidate_scores.append([global_config['distance_threshold'], global_config['is_alive_probability_weighting'], \
            global_config['positional_probabilities'], \
            current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind])
        logging.debug(str([current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind]))
        try:
            A = np.array(result_list)
            numpy.savetxt("experiments/" + global_config['experiment_series'] + "/hyperparameter_search.csv", A)
        except Exception:
            pass
        if current_score > best_score:
            best_score = current_score
            best_candidate = candidate
    global_config['distance_threshold'] = best_candidate

    logging.info('robustness test finished!')
    code.interact(local=dict(globals(), **locals()))

    pw = 1.0
    best_score = 0.0
    candidates = [0.0, 0.5 * pw, pw, 2.0 * pw]
    best_candidate = candidates[0]
    for candidate in candidates:
        logging.debug('run distance_threshhold %s', str(candidate))
        global_config['is_alive_probability_weighting'] = candidate
        current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(global_config, \
            experiment_series_names=global_config['experiment_series'])
        #
        candidate_scores.append([global_config['distance_threshold'], global_config['is_alive_probability_weighting'], \
            global_config['positional_probabilities'], \
            current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind])
        logging.debug(str([current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind]))
        try:
            A = np.array(result_list)
            numpy.savetxt("experiments/" + global_config['experiment_series'] + "/hyperparameter_search.csv", A)
        except Exception:
            pass
        if current_score > best_score:
            best_score = current_score
            best_candidate = candidate
    global_config['is_alive_probability_weighting'] = best_candidate

    pw = 1.0
    best_score = 0.0
    candidates = [0.0, 0.5 * pw, pw, 2.0 * pw]
    best_candidate = candidates[0]
    for candidate in candidates:
        logging.debug('run positional_probabilities %s', str(candidate))
        global_config['positional_probabilities'] = candidate
        current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(global_config, \
            experiment_series_names=global_config['experiment_series'])
        #
        candidate_scores.append([global_config['distance_threshold'], global_config['is_alive_probability_weighting'], \
            global_config['positional_probabilities'], \
            current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind])
        logging.debug(str([current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind]))
        try:
            A = np.array(result_list)
            numpy.savetxt("experiments/" + global_config['experiment_series'] + "/hyperparameter_search.csv", A)
        except Exception:
            pass
        if current_score > best_score:
            best_score = current_score
            best_candidate = candidate
    global_config['positional_probabilities'] = best_candidate

    try:
        A = np.array(result_list)
        numpy.savetxt("experiments/" + global_config['experiment_series'] + "/hyperparameter_search.csv", A)
    except Exception:
        pass

logging.info('data association finished!')
code.interact(local=dict(globals(), **locals()))
