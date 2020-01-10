import code  # code.interact(local=dict(globals(), **locals()))
import moviepy
from moviepy.editor import ImageSequenceClip
import copy, shutil, os

from data_association import DataAssociation

global_config = {
    'is_loaded': True,
    'weights_path': 'models/my_16_16_rnn.h5',
    'model_path': 'models/rnn_model_fake_data.h5',
    'distance_threshhold': 0.03,  # 5.0 / 2000,
    'num_timesteps': 50000,  # TODO
    #
    'initial_is_alive_probability': 0.5,
    'is_alive_decrease': 0.25,
    'is_alive_increase': 0.5,
    'batch_size': 64,  # TODO
    #
    'dataset_type': 'FakeDataset',
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
