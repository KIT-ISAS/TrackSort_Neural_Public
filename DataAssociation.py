import numpy as np

from TrackManager import TrackManager
from DataManager import DataManager

class DataAssociation(object):
	def __init__(self, global_config):
		self.global_config = global_config
		self.data_source = DataManager(global_config)
		self.track_manager = TrackManager(global_config, self.data_source)

	def associate_data(self):
		for time_step in range(self.global_config['max_num_steps']):
			#
			measurements = dataset.get_measurement_at_timestep(time_step)
			#
			predictions = self.track_manager.get_predictions()
			prediction_ids = predictions.keys()
			prediction_values = predictions.values()
			#
			distance_matrix = np.inf * np.ones([2 * len(measurements) + len(predictions), 2 * len(predictions) + len(measurements)])
			#
			for measurement_nr in range(len(measurements)):
				for prediction_nr in range(len(prediction_values)):
					distance_matrix[measurement_nr][prediction_nr] = np.linalg.norm(measurements[measurement_nr] - prediction_values[prediction_nr])
			#
			for measurement_nr in range(len(measurements)):
				distance_matrix[measurement_nr][len(prediction_values) + measurement_nr] = self.global_config['distance_threshhold']
				distance_matrix[len(measurements) + len(prediction_values) + measurement_nr][len(prediction_values) + measurement_nr] = self.global_config['distance_threshhold']
			#
			for prediction_nr in range(len(prediction_values)):
				distance_matrix[len(measurements) + prediction_nr][prediction_nr] = self.global_config['distance_threshhold']
				distance_matrix[len(measurements) + prediction_nr][len(measurements) + len(prediction_values) + prediction_nr] = self.global_config['distance_threshhold']
			#
			measurement_idxs, prediction_idxs = linear_sum_assignment(distance_matrix)
			#
			for idx in range(len(measurement_idxs)):
				if measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] < len(prediction_values):
					#
					model_manager.real_track_real_measurement(prediction_ids[prediction_idxs[idx]], measurements[measurement_idxs[idx]])
					#
				elif measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] < len(prediction_values):
					# feed it back its own prediction as measurement
					model_manager.real_track_pseudo_measurement(prediction_ids[prediction_idxs[idx]], prediction_values[prediction_idxs[idx]])
					#
				elif measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] < len(prediction_values):
					#
					model_manager.pseudo_track_real_measurement(measurements[measurement_idxs[idx]], time_step)