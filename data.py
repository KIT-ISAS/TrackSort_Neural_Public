# data.py
# -------

import random
import numpy as np

from abc import ABC, abstractmethod


class AbstractDataSet(ABC):
#	@abstractmethod
	def get_seq2seq_data(self, nan_value=0):
		"""
		Return a numpy array which can be used for RNN training.
		All trajectories start at timestep 0 and end at timestep max_timestep.
		max_timestep is the length of the longest trajectory. All other
		trajectories are padded with nan_value.

		shape: [number_tracks, max_timestep, point_dimension * 2]
		"""
		raise NotImplementedError

#	@abstractmethod
	def get_mlp_data(self):
		"""
		Return a numpy array which can be used for MLP Training.
		The input are n datapoints and the label is the next datapoint which
		should be predicted.
		Example: [ [[0,0], [0,1], [0,2], [0,3]],   [0,4] ]
		"""
		raise NotImplementedError

#	@abstractmethod
	def get_association_data(self):
		"""
		Return a numpy array with shape: [number_tracks, total_number_timesteps, point_dimension]
		total_number_timesteps is the timestep when the last measurement was made.

		This tensor differs from the others as the trajectories don't have to start
		at timestep=0.

		Attention: The trajectories are not sorted.

		Example data structure:
		          time | traj1 | traj2 | traj3
			--------------------------------
			     0 | (1,2) | NaN   | NaN
                             1 | (2,3) | (0,0) | NaN
                             2 | (3,4) | (1,0) | (0,0)
                             3 | NaN   | (2,0) | (1,1)
                             4 | NaN   | NaN   | (2,2)
		"""
		raise NotImplementedError

	def get_measurement_at_timestep(self, timestep):
		"""
		Return a numpy array with shape: [number_points, point_dimension]

		This is the same as: 
			>>> data = self.get_association_data()[:, timestep, :]
			>>> data[~numpy.isnan(data)]
		"""
		raise NotImplementedError



class FakeDataSet(AbstractDataSet):
	def __init__(self, timesteps=100, number_trajectories=10, step_length=10, noise_stddev=0, splits=0, belt_width=2000, belt_height=2000):
		"""
		Create Fake Data Lines for timesteps with normally distributed noise on a belt.

		@param timesteps: how many timesteps should the whole measurement take
		@param number_trajectories: how many trajectories to generate
		@param step_length: how long should the gap between sequential measurements be
		@param noise_stddev: add normally distributed noise with the given stddev in all directions [unit: pixels]
		@param splits: how many tracks should have missing measurements in between
		@param belt_width: width of the belt (left is zero)
		@param belt_height: height of the belt (bottom is zero)

		Attention: x-coordinate is along the height of the belt.
			   y-coordinate is along the width of the belt.
		"""
		self.timesteps = timesteps
		self.n_dim = 2
		self.n_trajectories = number_trajectories
		self.step_length = step_length
		self.belt_max_x = belt_height
		self.belt_max_y = belt_width

		self.track_data = self._generate_tracks()
		self.rnn_data = self._convert_tracks_to_rnn_data(self.track_data)

	def _generate_tracks(self):
		tracks = []
		# for every trajectory
		for track_number in range(self.n_trajectories):
			# pick a random point in time to start the trajectory
			start_timestep = random.randint(0, self.timesteps)

			# list containing all points: add NaNs for every skipped timestep
			trajectory = [np.NaN for _ in range(start_timestep)]

			# spawn the new trajectory on the left side
			new_y = 0
			new_x = random.randint(0, self.belt_max_x)
			trajectory.append((new_x, new_y))

			# iterate over all the timesteps from start_timestep+1   until end
			for t in range(start_timestep+2, self.timesteps+1):
				# generate next position
				trajectory.append((new_x, trajectory[-1][1] + self.step_length))

			tracks.append(trajectory)

		return np.array(tracks)

	def _convert_tracks_to_rnn_data(self, track_data):
		return None


if __name__ == '__main__':
	f = FakeDataSet()
	print(f.track_data.shape)
	print(f.track_data)
