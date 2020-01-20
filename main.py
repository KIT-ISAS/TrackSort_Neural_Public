import code  # code.interact(local=dict(globals(), **locals()))
import shutil
import argparse

from moviepy.editor import ImageSequenceClip
from data_association import DataAssociation
from evaluator import Evaluator

parser = argparse.ArgumentParser()

# the possible arguments you can give to the model
parser.add_argument('--is_loaded', type=bool, default=True, help='Whether the model is loaded or created + trained.')
parser.add_argument('--model_path', default='models/rnn_model_fake_data.h5', help='The path where the model is stored or loaded from.')
parser.add_argument('--matching_algorithm', default='global', help='The algorithm, that is used for matching. Current options are: ["local","global"])')
parser.add_argument('--dataset_dir', default='data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv', help='The directory of the data set. Only needed for CsvDataset.')
parser.add_argument('--dataset_type', default='CsvDataset', help='The type of the dataset. Current options are: ["FakeDataset","CsvDataset"].')
parser.add_argument('--distance_threshold', type=float, default=0.01, help='The threshhold, that is used for the matching with the artificial measurements and predictions')
parser.add_argument('--batch_size', type=int, default=64, help='The batchsize, that is used for training and inference')
parser.add_argument('--num_timesteps', type=int, default=15, help='The number of timesteps of the dataset. Necessary for FakeDataset.')
parser.add_argument('--num_train_epochs', type=int, default=1000, help='Only necessary, when model is trained.')
parser.add_argument('--nan_value', type=float, default=0.0, help='The Nan value, that is used by the DataManager')
parser.add_argument('--birth_rate_mean', type=float, default=5.0, help='The birth_rate_mean value, that is used by the DataManager')
parser.add_argument('--birth_rate_std', type=float, default=2.0, help='The birth_rate_std value, that is used by the DataManager')
parser.add_argument('--min_number_detections', type=int, default=6, help='The min_number_detections value, that is used by the DataManager')
parser.add_argument('--input_dim', type=int, default=2, help='The input_dim value, that is used by the DataManager')
parser.add_argument('--data_is_aligned', type=bool, default=True, help='Whether the data used by the DataManger is aligned or not.')

args = parser.parse_args()

global_config = {
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
        'birth_rate_std': args.birth_rate_std
    },
    #
    'num_train_epochs': args.num_train_epochs,
    'visualization_path': 'visualizations/matching_visualization/',
    'best_visualization_path': 'visualizations/best_matching_visualization/',
    'visualization_video_path': 'visualizations/matching_visualization_vid.mp4',
    'best_visualization_video_path': 'visualizations/best_matching_visualization_vid.mp4',
    'state_overwriting_started': False,
    'overwriting_activated': False,
    'verbos': 0,
    'visualize' : False,
    'run_hyperparameter_search' : True,
    'debug' : False
}


def run_global_config(global_config):
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
	score = 2 * accuracy_of_the_first_kind * accuracy_of_the_second_kind / (accuracy_of_the_first_kind + accuracy_of_the_second_kind)
	return score, accuracy_of_the_first_kind, accuracy_of_the_second_kind

dt = global_config['distance_threshold']
best_score = 0.0
distance_threshold_candidates = [0.25 * dt, 0.5 * dt, dt, 2.0 * dt, 4.0 * dt]
dtc_scores = []
best_dtc = distance_threshold_candidates[0]
for dtc in distance_threshold_candidates:
	print('run distance_threshhold ' + str(dtc))
	global_config['distance_threshold'] = dtc
	current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind = run_global_config(global_config)
	dtc_scores.append([current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind])
	print(str([current_score, accuracy_of_the_first_kind, accuracy_of_the_second_kind]))
	if current_score > best_score:
		best_score = current_score
		best_dtc = dtc
		if global_config['visualize']:
			shutil.move(global_config['visualization_video_path'], global_config['best_visualization_video_path'])
			shutil.move(global_config['visualization_path'], global_config['best_visualization_path'])
global_config['distance_threshold'] = best_dtc


print('data association finished!')
code.interact(local=dict(globals(), **locals()))