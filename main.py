import code  # code.interact(local=dict(globals(), **locals()))
import shutil
import argparse

from moviepy.editor import ImageSequenceClip
from data_association import DataAssociation

parser = argparse.ArgumentParser()

# the possible arguments you can give to the model
parser.add_argument('--is_loaded', type=bool, default=True, help='Whether the model is loaded or created + trained.')
parser.add_argument('--model_path', default='models/rnn_model_fake_data.h5', help='The path where the model is stored or loaded from.')
parser.add_argument('--matching_algorithm', default='local', help='The algorithm, that is used for matching. Current options are: ["local","global"])')
parser.add_argument('--dataset_dir', default='data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv', help='The directory of the data set. Only needed for CsvDataset.')
parser.add_argument('--dataset_type', default='CsvDataset', help='The type of the dataset. Current options are: ["FakeDataset","CsvDataset"].')
parser.add_argument('--distance_threshold', type=float, default=0.03, help='The threshhold, that is used for the matching with the artificial measurements and predictions')
parser.add_argument('--batch_size', type=int, default=64, help='The batchsize, that is used for training and inference')
parser.add_argument('--num_timesteps', type=int, default=359, help='The number of timesteps of the dataset. Necessary for FakeDataset.')
parser.add_argument('--num_train_epochs', type=int, default=1000, help='Only necessary, when model is trained.')
parser.add_argument('--nan_value', type=float, default=0.0, help='The Nan value, that is used by the DataManager')
parser.add_argument('--birth_rate_mean', type=float, default=1.0, help='The birth_rate_mean value, that is used by the DataManager')
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
    'visualization_path': 'visualizations/matching_visualization_local/',
    'visualization_video_path': 'visualizations/matching_visualization_vid.mp4',
    'state_overwriting_started': False,
    'overwriting_activated': False
}

data_association = DataAssociation(global_config)

tracks = data_association.associate_data()
particles = data_association.data_source.get_particles()

shutil.rmtree(global_config['visualization_video_path'], ignore_errors=True)
clip = ImageSequenceClip(global_config['visualization_path'], fps=4)
clip.write_videofile(global_config['visualization_video_path'], fps=4)

print('data association finished!')
code.interact(local=dict(globals(), **locals()))


# type of errors
# error of first kind: track contains multiple particles
def assigment_of_measurement_in_particle(timestep, measurement):
    for idx, particle_list in enumerate(particles):
        for particle in particle_list:
            particle_timestep, particle_measurement = particle
            if particle_timestep == timestep and particle_measurement[0] == measurement[0] and particle_measurement[1] == measurement[1]:
                return idx
    else:
        print('measurement found no match in particles!')
        code.interact(local=dict(globals(), **locals()))
#
num_errors_of_first_kind = 0
for track_id, track in enumerate(tracks):
    particle_id = assigment_of_measurement_in_particle(track.initial_timestep, track.measurements[0])
    correct_condition = lambda x: assigment_of_measurement_in_particle(x[0], x[1]) != particle_id
    check_list = []
    for it, measurement in enumerate(track.measurements):
        check_list.append([track.initial_timestep + it, measurement])
    num_errors_of_first_kind += int(len(list(filter(correct_condition, check_list))) != 0)

ratio_error_of_first_kind = num_errors_of_first_kind / len(tracks)
print('ratio_error_of_first_kind: ' + str(ratio_error_of_first_kind))

# type of errors
# error of first kind: track contains multiple particles
def assigment_of_measurement_in_track(timestep, measurement):
    for idx, track in enumerate(tracks):
        for it, track_measurement in enumerate(track.measurements):
            track_timestep = track.initial_timestep + it
            if track_timestep == timestep and track_measurement[0] == measurement[0] and track_measurement[1] == measurement[1]:
                return idx
    else:
        print('measurement found no match in tracks!')
        code.interact(local=dict(globals(), **locals()))
#
num_errors_of_second_kind = 0
for particle_id, particle_list in enumerate(particles):
    track_id = assigment_of_measurement_in_track(particle_list[0][0], particle_list[0][1])
    correct_condition = lambda x: assigment_of_measurement_in_track(x[0], x[1]) != track_id
    check_list = []
    for it, particle in enumerate(particle_list):
        check_list.append([particle[0], particle[1]])
    num_errors_of_second_kind += int(len(list(filter(correct_condition, check_list))) != 0)

ratio_error_of_second_kind = num_errors_of_second_kind / len(tracks)
print('ratio_error_of_second_kind: ' + str(ratio_error_of_second_kind))

