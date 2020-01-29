import code  # code.interact(local=dict(globals(), **locals()))
import os
import shutil
import argparse
import json
import datetime
import logging

from moviepy.editor import ImageSequenceClip
from data_association import DataAssociation
from evaluator import Evaluator

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
parser.add_argument('--matching_algorithm', default='global',
                    help='The algorithm, that is used for matching. Current options are: ["local","global"])')
parser.add_argument('--dataset_dir', default='data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv',
                    help='The directory of the data set. Only needed for CsvDataset.')
parser.add_argument('--dataset_type', default='CsvDataset',
                    help='The type of the dataset. Current options are: ["FakeDataset","CsvDataset"].')
parser.add_argument('--distance_threshold', type=float, default=0.02,
                    help='The threshold used for the matching with the artificial measurements and predictions')
parser.add_argument('--batch_size', type=int, default=64, help='The batchsize, that is used for training and inference')
parser.add_argument('--num_timesteps', type=int, default=10000,
                    help='The number of timesteps of the dataset. Necessary for FakeDataset.')
parser.add_argument('--num_train_epochs', type=int, default=1000, help='Only necessary, when model is trained.')
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

parser.add_argument('--virtual_belt_edge_x_position', type=float, default=800,
                    help='Where does the virtual belt end?')
parser.add_argument('--virtual_nozzle_array_x_position', type=float, default=1550,
                    help='Where should the virtual nozzle array be?')
parser.add_argument('--num_units_first_rnn', type=int, default=1024,
                    help='Where should the virtual nozzle array be?')
parser.add_argument('--num_units_second_rnn', type=int, default=16,
                    help='Where should the virtual nozzle array be?')

args = parser.parse_args()

global_config = {
    'separation_prediction': args.separation_prediction,
    'time_normalization_constant': args.time_normalization_constant,
    'virtual_belt_edge_x_position': args.virtual_belt_edge_x_position,
    'virtual_nozzle_array_x_position': args.virtual_nozzle_array_x_position,
    'only_last_timestep_additional_loss': True,

    'is_loaded': args.is_loaded,
    'model_path': args.model_path,
    'distance_threshold': args.distance_threshold,
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
        'batch_size': args.batch_size,
        'data_is_aligned': args.data_is_aligned,
        'birth_rate_mean': args.birth_rate_mean,
        'birth_rate_std': args.birth_rate_std,
        'rotate_columns': args.rotate_columns,
        'normalization_constant': args.normalization_constant,
        'additive_noise_stddev': 0.0
    },
    'rnn_model_factory': {
        'num_units_first_rnn': args.num_units_first_rnn,
        'num_units_second_rnn': args.num_units_second_rnn,
        'num_units_third_rnn': 0,
        'num_units_fourth_rnn': 0,
        'num_units_first_dense': 0,
        'num_units_second_dense': 0,
        'num_units_third_dense': 0,
        'num_units_fourth_dense': 0,
        'rnn_model_name': 'lstm',
        'use_batchnorm_on_dense': True,
    },
    #
    'num_train_epochs': args.num_train_epochs,
    'evaluate_every_n_epochs': args.evaluate_every_n_epochs,
    'lr_decay_after_epochs': args.lr_decay_after_epochs,
    'lr_decay_factor': args.lr_decay_factor,

    'state_overwriting_started': False,
    'overwriting_activated': False,
    'verbose': 1,
    'visualize': True,
    'run_hyperparameter_search': args.run_hyperparameter_search,
    'debug': False,
    'is_alive_probability_weighting': 1.0,
    'test_noise_robustness': args.test_noise_robustness
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


def run_global_config(global_config):
    # Create paths for run
    now = datetime.datetime.now()
    experiment_name = now.strftime("%Y_%m_%d__%H_%M_%S")
    global_config['experiment_name'] = experiment_name
    global_config['diagrams_path'] = os.path.join('visualizations', experiment_name, 'diagrams')
    os.makedirs(global_config['diagrams_path'])
    global_config['visualization_path'] = 'visualizations/' + experiment_name + '/matching_visualization/'
    global_config['visualization_video_path'] = 'visualizations/' + experiment_name + '/matching_visualization_vid.mp4'
    #
    data_association = DataAssociation(global_config)
    particles = data_association.data_source.get_particles()
    tracks = data_association.associate_data()
    #
    if global_config['visualize']:
        shutil.rmtree(global_config['visualization_video_path'], ignore_errors=True)
        clip = ImageSequenceClip(global_config['visualization_path'], fps=4)
        clip.write_videofile(global_config['visualization_video_path'], fps=4)
    #
    evaluator = Evaluator(global_config, particles, tracks)
    accuracy_of_the_first_kind = 1.0 - evaluator.error_of_first_kind()
    accuracy_of_the_second_kind = 1.0 - evaluator.error_of_second_kind()
    score = 2 * accuracy_of_the_first_kind * accuracy_of_the_second_kind / (
            accuracy_of_the_first_kind + accuracy_of_the_second_kind)
    # save the current config
    global_config['score'] = score
    global_config['accuracy_of_the_first_kind'] = accuracy_of_the_first_kind
    global_config['accuracy_of_the_second_kind'] = accuracy_of_the_second_kind
    with open('experiments/' + global_config['experiment_name'], 'w') as file_:
        json.dump(global_config, file_, sort_keys=True, indent=4)
    #
    return score, accuracy_of_the_first_kind, accuracy_of_the_second_kind


if not global_config['run_hyperparameter_search']:
    if not global_config['test_noise_robustness']:
        score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(global_config)
        logging.info('data association finished!')
        code.interact(local=dict(globals(), **locals()))
    else:
        logging.info('test robustness against noise!')
        result_list = []
        for noise in [0.0, 0.0003, 0.0005, 0.0008, 0.001]:
            global_config['CsvDataSet']['additive_noise_stddev'] = noise
            score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(global_config)
            result_list.append([score, accuracy_of_the_first_kind, accuracy_of_the_second_kind])
        logging.debug(str(result_list))
        logging.info('robustness test finished!')
        code.interact(local=dict(globals(), **locals()))

else:
    dt = global_config['distance_threshold']
    best_score = 0.0
    distance_threshold_candidates = [0.25 * dt, 0.5 * dt, dt, 2.0 * dt, 4.0 * dt]
    dtc_scores = []
    best_dtc = distance_threshold_candidates[0]
    for dtc in distance_threshold_candidates:
        logging.debug('run distance_threshhold %s', str(dtc))
        global_config['distance_threshold'] = dtc
        current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(global_config)
        #
        dtc_scores.append([current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind])
        logging.debug(str([current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind]))
        if current_score > best_score:
            best_score = current_score
            best_dtc = dtc
    global_config['distance_threshold'] = best_dtc

    pw = global_config['is_alive_probability_weighting']
    best_score = 0.0
    pb_candidates = [0.25 * pw, 0.5 * pw, pw, 2.0 * pw, 4.0 * pw]
    best_dtc = pb_candidates[0]
    for dtc in pb_candidates:
        logging.debug('run distance_threshhold %s', str(dtc))
        global_config['distance_threshold'] = dtc
        current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(global_config)
        #
        dtc_scores.append([current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind])
        logging.debug(str([current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind]))
        if current_score > best_score:
            best_score = current_score
            best_dtc = dtc
    global_config['distance_threshold'] = best_dtc

logging.info('data association finished!')
code.interact(local=dict(globals(), **locals()))
