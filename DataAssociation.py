import numpy as np
from scipy.optimize import linear_sum_assignment
import code # code.interact(local=dict(globals(), **locals()))

from TrackManager import TrackManager
from DataManager import FakeDataSet

class DataAssociation(object):
	def __init__(self, global_config):
		self.global_config = global_config
		if self.global_config['dataset_type'] == 'FakeDataset':
			self.data_source = FakeDataSet(global_config=global_config)
		else:
			self.data_source = CsvDataSet(global_config=global_config)
		self.track_manager = TrackManager(global_config, self.data_source)

	def associate_data(self):
		for time_step in range(self.global_config['num_timesteps']):
			print('step ' + str(time_step) + ' / ' + str(self.global_config['num_timesteps']))
			#
			measurements = self.data_source.get_measurement_at_timestep_list(time_step)
			#
			predictions = self.track_manager.get_predictions()
			prediction_ids = list(predictions.keys())
			prediction_values = list(predictions.values())
			#
			distance_matrix = np.inf * np.ones([2 * len(measurements) + len(predictions), 2 * len(predictions) + len(measurements)])
			#
			for measurement_nr in range(len(measurements)):
				for prediction_nr in range(len(prediction_values)):
					distance_matrix[measurement_nr][prediction_nr] = np.linalg.norm(measurements[measurement_nr] - prediction_values[prediction_nr])
			#
			for measurement_nr in range(len(measurements)):
				distance_matrix[measurement_nr][len(prediction_values) + measurement_nr] = self.global_config['distance_threshhold']
				distance_matrix[len(measurements) + len(prediction_values) + measurement_nr][len(prediction_values) + measurement_nr] = 1.1 * self.global_config['distance_threshhold']
			#
			for prediction_nr in range(len(prediction_values)):
				distance_matrix[len(measurements) + prediction_nr][prediction_nr] = self.global_config['distance_threshhold']
				distance_matrix[len(measurements) + prediction_nr][len(measurements) + len(prediction_values) + prediction_nr] = 1.1 * self.global_config['distance_threshhold']
			#
			measurement_idxs, prediction_idxs = linear_sum_assignment(distance_matrix)
			#
			counts = np.zeros([4], dtype=np.int32)
			for idx in range(len(measurement_idxs)):
				if measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] < len(prediction_values):
					#print('realreal')
					counts[0] += 1
					#
					self.track_manager.real_track_real_measurement(prediction_ids[prediction_idxs[idx]], measurements[measurement_idxs[idx]])
					#
				elif measurement_idxs[idx] >= len(measurements) and prediction_idxs[idx] < len(prediction_values):
					#print('realpseudo')
					counts[1] += 1
					# feed it back its own prediction as measurement
					self.track_manager.real_track_pseudo_measurement(prediction_ids[prediction_idxs[idx]], prediction_values[prediction_idxs[idx]])
					#
				elif measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] >= len(prediction_values):
					#print('pseudoreal')
					counts[2] += 1
					#
					self.track_manager.pseudo_track_real_measurement(measurements[measurement_idxs[idx]], time_step)
				else:
					#print('pseudopseudo')
					counts[3] += 1
			print(list(counts))
			#print('in associate_data: ' + str(time_step))
			#code.interact(local=dict(globals(), **locals()))
		#
		return self.track_manager.tracks