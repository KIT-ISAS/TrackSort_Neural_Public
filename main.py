import code  # code.interact(local=dict(globals(), **locals()))
import shutil

from moviepy.editor import ImageSequenceClip
from data_association import DataAssociation

batch_size = 64

global_config = {
    'is_loaded': True,
    'weights_path': 'models/my_16_16_rnn.h5',
    'model_path': 'models/rnn_model_fake_data.h5',
    'distance_threshold': 0.03,  # 5.0 / 2000,
    'batch_size': batch_size,
    #
    'Track': {
        'initial_is_alive_probability': 0.5,
        'is_alive_decrease': 0.25,
        'is_alive_increase': 0.5,
    },

    'num_timesteps': 50,
    'dataset_type': 'CsvDataset',
    #
    'CsvDataSet': {
        'glob_file_pattern': 'data/Pfeffer/trackSortResultPfeffer/*_trackHistory_NothingDeleted.csv',
        'min_number_detections': 6,
        'nan_value': 0,
        'input_dim': 2,
        'batch_size': batch_size,
        'data_is_aligned': True,
        'birth_rate_mean': 1,
        'birth_rate_std': 2
    },
    #
    'num_train_epochs': 1000,
    'visualization_path': 'visualizations/matching_visualization_local/',
    'visualization_video_path': 'visualizations/matching_visualization_vid.mp4',
    'state_overwriting_started': False,
    'overwriting_activated': False
}

data_association = DataAssociation(global_config)

tracks = data_association.associate_data()

shutil.rmtree(global_config['visualization_video_path'], ignore_errors=True)
clip = ImageSequenceClip(global_config['visualization_path'], fps=4)
clip.write_videofile(global_config['visualization_video_path'], fps=4)

print('data association finished!')
code.interact(local=dict(globals(), **locals()))
