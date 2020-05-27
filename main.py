#!/usr/bin/env python

import code  # code.interact(local=dict(globals(), **locals()))
import os
import shutil
import argparse
import json
import datetime
import logging

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import ImageSequenceClip
from track_manager import TrackManager
from model_manager import ModelManager
from data_association import DataAssociation
from data_manager import FakeDataSet, CsvDataSet
from evaluation_functions import calculate_error_first_and_second_kind
from kalman_playground import kalman_playground
from velocity_plot import velocity_plot
# Test
from cv_model import *
from ca_model import *

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
                    help='Whether the models should be loaded or trained.')
parser.add_argument('--is_loaded_gating_network', type=str2bool, default=True,
                    help='Whether the gating network should be loaded or trained.')
parser.add_argument('--model_path', default='models/rnn_model_fake_data.h5',
                    help='The path where the model is stored or loaded from.')
parser.add_argument('--matching_algorithm', default='global', choices=['local', 'global'],
                    help='The algorithm, that is used for matching.')
parser.add_argument('--dataset_dir', default='data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv',
                    help='The directory of the data set. Only needed for CsvDataset.')
parser.add_argument('--dataset_type', default='CsvDataset', choices=['FakeDataset', 'CsvDataset'],
                    help='The type of the dataset.')
parser.add_argument('--result_path', default='results/default_results/',
                    help='The path where the model is stored or loaded from.')
parser.add_argument('--distance_threshold', type=float, default=0.02,
                    help='The threshold used for the matching with the artificial measurements and predictions')
parser.add_argument('--config_path', default="configs/default_config.json",
                    help='Path to config file including information about experts, gating network and weighting function.')
parser.add_argument('--batch_size', type=int, default=64, help='The batchsize, that is used for training and inference')
parser.add_argument('--evaluation_ratio', type=float, default=0.15, help='The ratio of data used for evaluation.')
parser.add_argument('--test_ratio', type=float, default=0.15, help='The ratio of data used for the final unbiased test.')
parser.add_argument('--num_timesteps', type=int, default=350,
                    help='The number of timesteps of the dataset. Necessary for FakeDataset.')
parser.add_argument('--num_train_epochs', type=int, default=1000, help='Only necessary, when model is trained.')
parser.add_argument('--improvement_break_condition', type=float, default=-100, 
                    help='Break training if test loss on every expert does not improve by more than this value.')
parser.add_argument('--lr_decay_after_epochs', type=int, default=150, help='When to decrease the lr by lr_decay_factor')
parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='When learning rate should be decreased, '
                                                                       'multiply with this')
parser.add_argument('--nan_value', type=float, default=0.0, help='The Nan value, that is used by the DataManager')
parser.add_argument('--birth_rate_mean', type=float, default=5.0,
                    help='The birth_rate_mean value, that is used by the DataManager')
parser.add_argument('--birth_rate_std', type=float, default=2.0,
                    help='The birth_rate_std value, that is used by the DataManager')
parser.add_argument('--normalization_constant', type=float, default=None, help='Normalization value')
parser.add_argument('--evaluate_every_n_epochs', type=int, default=50)
parser.add_argument('--time_normalization_constant', type=float, default=22.0, help='Normalization for time prediction')
parser.add_argument('--input_dim', type=int, default=2, help='The input_dim value, that is used by the DataManager')
parser.add_argument('--mlp_input_dim', type=int, default=5, help='The dimension of input points for the MLP')
parser.add_argument('--separation_mlp_input_dim', type=int, default=7, help='The dimension of input points for the separation MLP')
parser.add_argument('--data_is_aligned', type=str2bool, default=True,
                    help='Whether the data used by the DataManger is aligned or not.')
parser.add_argument('--rotate_columns', type=str2bool, default=False,
                    help='Set this to true if the order of columns in your csv is (x, y). Default is (y, x)')
parser.add_argument('--run_hyperparameter_search', type=str2bool, default=False,
                    help='Whether to run the hyperparameter search or not')
parser.add_argument('--test_noise_robustness', type=str2bool, default=False,
                    help='Should the Dataset be tested with multiple noise values?')
parser.add_argument('--tracking', type=str2bool, default=True,
                    help='Perform tracking. This is the default mode.')
parser.add_argument('--separation_prediction', type=str2bool, default=False,
                    help='Perform separation predcition')
parser.add_argument('--verbosity', default='INFO', choices=logging._nameToLevel.keys())

parser.add_argument('--additive_noise_stddev', type=float, default=0.0)

parser.add_argument('--virtual_belt_edge_x_position', type=float, default=800,
                    help='Where does the virtual belt end?')
parser.add_argument('--virtual_nozzle_array_x_position', type=float, default=1550,
                    help='Where should the virtual nozzle array be?')
parser.add_argument('--min_measurements_count', type=int, default=3,
                    help='Ignore tracks with less measurements.')                    

parser.add_argument('--clear_state', type=str2bool, default=True,
                    help='Whether a new track should be initialized with empty state?')
parser.add_argument('--overwriting_activated', type=str2bool, default=True,
                    help='Whether batches of the RNN are reused')
parser.add_argument('--execute_evaluation', type=str2bool, default=True,
                    help='Run evaluation after training/loading or not')
parser.add_argument('--execute_multi_target_tracking', type=str2bool, default=True,
                    help='Run multi-target tracking after training/loading or not')
parser.add_argument('--evaluate_mlp_mask', type=str2bool, default=False,
                    help='Masks every model with a mlp masks to compare MLPs with other models in testing function.')
parser.add_argument('--no_show', type=str2bool, default=False,
                    help='Set this to True if you do not want to show evaluation graphics and only save them.')
parser.add_argument('--visualize_multi_target_tracking', type=str2bool, default=False,
                    help='You can generate nice videos of the multitarget tracking.')
args = parser.parse_args()

global_config = {
    'separation_prediction': args.separation_prediction,
    'tracking': args.tracking,
    'clear_state': args.clear_state,
    'time_normalization_constant': args.time_normalization_constant,
    'virtual_belt_edge_x_position': args.virtual_belt_edge_x_position,
    'virtual_nozzle_array_x_position': args.virtual_nozzle_array_x_position,
    'only_last_timestep_additional_loss': True,

    'is_loaded': args.is_loaded,
    'is_loaded_gating_network': args.is_loaded_gating_network,
    'model_path': args.model_path,
    'result_path': args.result_path,
    'distance_threshold': args.distance_threshold,
    'config_path': args.config_path,
    'batch_size': args.batch_size,
    'evaluation_ratio': args.evaluation_ratio,
    'test_ratio': args.test_ratio,
    'matching_algorithm': args.matching_algorithm,
    #
    'Track': {
        'initial_is_alive_probability': 0.5,
        'is_alive_decrease': 0.25,
        'is_alive_increase': 0.5
    },
    #
    'num_timesteps': args.num_timesteps,
    'dataset_type': args.dataset_type,
    'mlp_input_dim': args.mlp_input_dim,
    #
    'CsvDataSet': {
        'glob_file_pattern': args.dataset_dir,
        'nan_value': args.nan_value,
        'input_dim': args.input_dim,
        'mlp_input_dim': args.mlp_input_dim,
        'data_is_aligned': args.data_is_aligned,
        'birth_rate_mean': args.birth_rate_mean,
        'birth_rate_std': args.birth_rate_std,
        'rotate_columns': args.rotate_columns,
        'normalization_constant': args.normalization_constant,
        'virtual_belt_edge_x_position': args.virtual_belt_edge_x_position,
        'virtual_nozzle_array_x_position': args.virtual_nozzle_array_x_position, 
        'min_measurements_count': args.min_measurements_count,
        'additive_noise_stddev': args.additive_noise_stddev,
        'is_separation_prediction': args.separation_prediction
    },
    'separation_mlp_input_dim': args.separation_mlp_input_dim,

    'num_train_epochs': args.num_train_epochs,
    'evaluate_every_n_epochs': args.evaluate_every_n_epochs,
    'improvement_break_condition': args.improvement_break_condition,
    'lr_decay_after_epochs': args.lr_decay_after_epochs,
    'lr_decay_factor': args.lr_decay_factor,

    'state_overwriting_started': False,
    'overwriting_activated': args.overwriting_activated,
    'verbose': 1,
    'visualize': args.visualize_multi_target_tracking,
    'run_hyperparameter_search': args.run_hyperparameter_search,
    'debug': False,
    'test_noise_robustness': args.test_noise_robustness,
    'experiment_series': 'independent',
    'is_alive_probability_weighting': 0.0,
    'positional_probabilities': 0.0,

    'execute_evaluation': args.execute_evaluation,
    'execute_multi_target_tracking': args.execute_multi_target_tracking,

    'evaluate_mlp_mask': args.evaluate_mlp_mask,
    'no_show':args.no_show
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

    #for config_key in ['results_path', 'experiment_path', 'diagrams_path', 'visualization_path']:
        #os.makedirs(global_config[config_key], exist_ok=True)

    # file paths
    global_config['visualization_video_path'] = os.path.join(global_config['visualization_path'],
                                                             'matching_visualization_vid.mp4')
    global_config['json_file'] = os.path.join(global_config['result_path'], 'config.json')

    ## Import data
    if global_config['dataset_type'] == 'FakeDataset':
        # TODO: Implement MLP dataset on fake data!
        data_source = FakeDataSet(mlp_input_dim=global_config['mlp_input_dim'], timesteps=global_config['num_timesteps'], batch_size=global_config['batch_size'])
    elif global_config['dataset_type'] == 'CsvDataset':
        data_source = CsvDataSet(**global_config['CsvDataSet'])
        global_config['num_timesteps'] = data_source.get_num_timesteps()
    
    # PLAY AROUND WITH VARIOUS KALMAN FILTERS
    # should be commented out...
    kalman_playground(data_source.aligned_track_data, data_source.normalization_constant)
    #velocity_plot(data_source.aligned_track_data, data_source.normalization_constant)

    ## Import model config to json tree
    # TODO: Create json schema to check config validity
    with open(global_config["config_path"]) as f:
        model_config = json.load(f)
        
    ## Initialize models
    # model_config, is_loaded, num_time_steps, overwriting_activated=True, x_pred_to = 1550, time_normalization = 22.
    model_manager = ModelManager(model_config = model_config,
                                 is_loaded = global_config.get("is_loaded"), 
                                 num_time_steps = data_source.longest_track, 
                                 n_mlp_features = global_config.get("mlp_input_dim"),
                                 n_mlp_features_separation_prediction = global_config.get("separation_mlp_input_dim"),
                                 overwriting_activated = global_config.get("overwriting_activated"),
                                 x_pred_to = global_config.get("CsvDataSet").get("virtual_nozzle_array_x_position")/data_source.normalization_constant,
                                 time_normalization = global_config['time_normalization_constant'])

    ## Get tracking training and test dataset
    # TODO: 
    #   * Ask for these arguments in main run args
    random_seed = 0
    if global_config["tracking"]:
        mlp_dataset_train, mlp_dataset_eval, mlp_dataset_test = data_source.get_tf_data_sets_mlp_data(
                                        normalized=True, 
                                        evaluation_ratio = global_config.get("evaluation_ratio"), 
                                        test_ratio= global_config.get("test_ratio"),
                                        batch_size = global_config.get('batch_size'), 
                                        random_seed = random_seed)

        seq2seq_dataset_train, seq2seq_dataset_eval, seq2seq_dataset_test = data_source.get_tf_data_sets_seq2seq_data(
                                        normalized=True, 
                                        evaluation_ratio = global_config.get("evaluation_ratio"), 
                                        test_ratio= global_config.get("test_ratio"),
                                        batch_size = global_config.get('batch_size'), 
                                        random_seed = random_seed)
    
    ## Get separation prediction training and test dataset
    if global_config["separation_prediction"]:
        mlp_dataset_train_sp, mlp_dataset_eval_sp, mlp_dataset_test_sp = \
            data_source.get_tf_data_sets_mlp_with_separation_data( 
                normalized=True, 
                evaluation_ratio = global_config.get("evaluation_ratio"), 
                test_ratio= global_config.get("test_ratio"),
                batch_size=global_config['batch_size'], 
                random_seed=random_seed,
                time_normalization=global_config['time_normalization_constant'],
                n_inp_points = global_config['separation_mlp_input_dim'])

        seq2seq_dataset_train_sp, seq2seq_dataset_eval_sp, seq2seq_dataset_test_sp, num_time_steps = \
            data_source.get_tf_data_sets_seq2seq_with_separation_data(
                normalized=True, 
                evaluation_ratio = global_config.get("evaluation_ratio"), 
                test_ratio= global_config.get("test_ratio"),
                batch_size=global_config['batch_size'], 
                random_seed=random_seed,
                time_normalization=global_config['time_normalization_constant'])
    ## Train models
    if not global_config["is_loaded"]:
        if global_config["tracking"]:
            model_manager.train_models(mlp_conversion_func = data_source.mlp_target_to_track_format,
                                    seq2seq_dataset_train = seq2seq_dataset_train,
                                    seq2seq_dataset_test = seq2seq_dataset_eval, 
                                    mlp_dataset_train = mlp_dataset_train,
                                    mlp_dataset_test = mlp_dataset_eval,
                                    num_train_epochs = global_config.get("num_train_epochs"),
                                    evaluate_every_n_epochs = global_config.get("evaluate_every_n_epochs"),
                                    improvement_break_condition = global_config.get("improvement_break_condition"),
                                    lr_decay_after_epochs = global_config.get("lr_decay_after_epochs"),
                                    lr_decay = global_config.get("lr_decay_factor"))
        if global_config["separation_prediction"]:
            model_manager.train_models_separation_prediction(seq2seq_dataset_train = seq2seq_dataset_train_sp,
                                    seq2seq_dataset_test = seq2seq_dataset_eval_sp, 
                                    mlp_dataset_train = mlp_dataset_train_sp,
                                    mlp_dataset_test = mlp_dataset_eval_sp,
                                    num_train_epochs = global_config.get("num_train_epochs"),
                                    evaluate_every_n_epochs = global_config.get("evaluate_every_n_epochs"),
                                    improvement_break_condition = global_config.get("improvement_break_condition"),
                                    lr_decay_after_epochs = global_config.get("lr_decay_after_epochs"),
                                    lr_decay = global_config.get("lr_decay_factor"))
    ## Train gating network     
    if global_config["tracking"]:                           
        if global_config["is_loaded_gating_network"]:
            model_manager.load_gating_network()
        else:
            model_manager.train_gating_network(mlp_conversion_func = data_source.mlp_target_to_track_format,
                                                seq2seq_dataset_train = seq2seq_dataset_train,
                                                mlp_dataset_train = mlp_dataset_train)
    if global_config["separation_prediction"]:
        stop=0
        # TODO: Implement separation prediction gating network

    ## Test models
    # TODO:
    #   * Test with an evaluation set instead of test set.
    if global_config["tracking"]:
        if global_config.get('execute_evaluation'):
            model_manager.test_models(mlp_conversion_func = data_source.mlp_target_to_track_format,
                                    result_dir = global_config['result_path'],
                                    seq2seq_dataset_test = seq2seq_dataset_eval, 
                                    mlp_dataset_test = mlp_dataset_eval,
                                    normalization_constant = data_source.normalization_constant,
                                    evaluate_mlp_mask = global_config['evaluate_mlp_mask'],
                                    no_show = global_config['no_show'])
    if global_config["separation_prediction"]:
        if global_config.get('execute_evaluation'):
            model_manager.test_models_separation_prediction(result_dir = global_config['result_path'],
                                    seq2seq_dataset_test = seq2seq_dataset_eval_sp, 
                                    mlp_dataset_test = mlp_dataset_eval_sp,
                                    normalization_constant = data_source.normalization_constant,
                                    time_normalization_constant=global_config['time_normalization_constant'],
                                    no_show = global_config['no_show'])
    ## Execute MTT
    if global_config.get('execute_multi_target_tracking'):
        # Check if result folder exists and create it if not.
        save_path = os.path.dirname(global_config['result_path'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ## Init tracks
        track_manager = TrackManager(model_config.get('data_association').get('track_config'))
        ## Get multi-target tracking data
        particle_time_list = data_source.get_particle_timestep_data()
        ## Run multi-target tracking
        data_association = DataAssociation(global_config.get('num_timesteps'), global_config.get('CsvDataSet').get('rotate_columns'),
                                        global_config.get("visualization_path"), global_config.get("visualize"),
                                        **model_config.get('data_association').get('association_config'))

        tracks = data_association.associate_data(particle_time_list, track_manager, model_manager, data_source.get_belt_limits())

        if global_config['visualize']:
            shutil.rmtree(global_config['visualization_video_path'], ignore_errors=True)
            clip = ImageSequenceClip(global_config['visualization_path'], fps=4)
            clip.write_videofile(global_config['visualization_video_path'], fps=4)

        # No ids are skipped, so this is can be used.
        particle_ids = np.arange(0, len(particle_time_list), 1)
        error_of_first_kind, error_of_second_kind = calculate_error_first_and_second_kind(tracks, particle_ids)
        accuracy_of_the_first_kind = 1.0 - error_of_first_kind
        accuracy_of_the_second_kind = 1.0 - error_of_second_kind
        score = 2 * accuracy_of_the_first_kind * accuracy_of_the_second_kind / (
                accuracy_of_the_first_kind + accuracy_of_the_second_kind)

        # save the current config
        global_config['score'] = score
        global_config['accuracy_of_the_first_kind'] = accuracy_of_the_first_kind
        global_config['accuracy_of_the_second_kind'] = accuracy_of_the_second_kind

        with open(global_config['json_file'], 'w') as file_:
            json.dump(global_config, file_, sort_keys=True, indent=4)

        del data_association
        del particle_time_list
        del tracks
        del global_config

        return score, accuracy_of_the_first_kind, accuracy_of_the_second_kind
    else:
        return 0.0, 0.0, 0.0


if not global_config['run_hyperparameter_search']:
    if not global_config['test_noise_robustness']:
        score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(global_config)
        logging.info('data association finished!')
        #code.interact(local=dict(globals(), **locals()))
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
        #code.interact(local=dict(globals(), **locals()))

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
            np.savetxt("experiments/" + global_config['experiment_series'] + "/hyperparameter_search.csv", A)
        except Exception:
            pass
        if current_score > best_score:
            best_score = current_score
            best_candidate = candidate
    global_config['distance_threshold'] = best_candidate

    logging.info('robustness test finished!')
    #code.interact(local=dict(globals(), **locals()))

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
            np.savetxt("experiments/" + global_config['experiment_series'] + "/hyperparameter_search.csv", A)
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
            np.savetxt("experiments/" + global_config['experiment_series'] + "/hyperparameter_search.csv", A)
        except Exception:
            pass
        if current_score > best_score:
            best_score = current_score
            best_candidate = candidate
    global_config['positional_probabilities'] = best_candidate

    try:
        A = np.array(result_list)
        np.savetxt("experiments/" + global_config['experiment_series'] + "/hyperparameter_search.csv", A)
    except Exception:
        pass

logging.info('data association finished!')
#code.interact(local=dict(globals(), **locals()))