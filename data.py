# data.py
# -------
# - AbstractDataSet: Abstract class which defines the necessary interface for every data set
#    - FakeDataSet: Randomly created lines with noise
#    - CsvDataSet: ToDo(Daniel): Loads (real) tracks from a glob pattern of *.csv files

import random
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class AbstractDataSet(ABC):
    belt_width = 2000
    belt_height = 2000

    nan_value = 0
    input_dim = 2

    @abstractmethod
    def get_seq2seq_data(self, nan_value=0):
        """
        Return a numpy array which can be used for RNN training.
        All trajectories start at timestep 0 and end at timestep max_timestep.
        max_timestep is the length of the longest trajectory. All other
        trajectories are padded with nan_value.

        shape: [number_tracks, max_timestep, point_dimension * 2]
        """
        raise NotImplementedError

    @abstractmethod
    def get_mlp_data(self):
        """
        Return a numpy array which can be used for MLP Training.
        The input are n datapoints and the label is the next datapoint which
        should be predicted.
        Example: [ [[0,0], [0,1], [0,2], [0,3]],   [0,4] ]
        """
        raise NotImplementedError

    @abstractmethod
    def get_track_data(self):
        """
        Return a numpy array with shape: [number_tracks, total_number_timesteps, point_dimension]
        total_number_timesteps is the timestep when the last measurement was made.

        This tensor differs from the others as the trajectories don't have to start
        at timestep=0.

        Attention: The trajectories are not sorted.

        Example data structure:
          time | traj1 | traj2 | traj3
          ----------------------------
             0 | (1,2) | NaN   | NaN
             1 | (2,3) | (0,0) | NaN
             2 | (3,4) | (1,0) | (0,0)
             3 | NaN   | (2,0) | (1,1)
             4 | NaN   | NaN   | (2,2)
        """
        raise NotImplementedError

    def get_aligned_track_data(self):
        """
        Return a numpy array with shape: [number_tracks, total_number_timesteps, point_dimension]
        total_number_timesteps is the timestep when the last measurement was made.

        The *alignment* signifies that all trajectories start at time_step=0!

        Attention: The trajectories are not sorted.

        Example data structure:
          time | traj1 | traj2 | traj3
          ----------------------------
             0 | (1,2) | (0,0) | (0,0)
             1 | (2,3) | (1,0) | (1,1)
             2 | (3,4) | (2,0) | (2,2)
             3 | (N,N) | (3,0) | (N,N)
             4 | (N,N) | (N,N) | (N,N)
        """
        raise NotImplementedError

    def get_measurement_at_timestep(self, timestep):
        """
        Return a numpy array with shape: [number_points, point_dimension]

        This is something like:
            >>> data = self.get_track_data()[:, timestep, :]
            >>> data[~np.isnan(data)]
        """
        raise NotImplementedError

    @classmethod
    def get_last_timestep_of_track(cls, track):
        i = None
        for i in range(track.shape[0]):
            if not np.any(track[i]):
                return i
        return i

    def plot_track(self, track, color='black', start=0, end=-1, label='track'):
        track = track[start:end]

        axes = plt.gca()
        axes.set_xlim([0, self.belt_width + 100])
        axes.set_ylim([0, self.belt_height + 100])

        plt.xlabel('x position [px]')
        plt.ylabel('y position [px]')

        axes.scatter(track[:, 0], track[:, 1], color=color)
        axes.plot(track[:, 0], track[:, 1], color=color, label=label)
        axes.legend()

        return axes

    def plot_random_tracks(self, n=15):
        tracks = self.get_aligned_track_data()
        for _ in range(n):
            random_index = random.randint(0, tracks.shape[0] - 1)
            random_track = tracks[random_index]
            axes = self.plot_track(random_track,
                                   color=np.random.rand(3),
                                   end=self.get_last_timestep_of_track(random_track),
                                   label='track {}'.format(random_index))
            plt.title('Tracks')
        plt.show()


class FakeDataSet(AbstractDataSet):
    def __init__(self, timesteps=35, number_trajectories=100,
                 additive_noise_stddev=5, splits=0, additive_target_stddev=100,
                 min_number_points_per_trajectory=10,
                 belt_width=2000, belt_height=2000):
        """
        Create Fake Data Lines for timesteps with normally distributed noise on a belt.

        @param timesteps: how many timesteps should the whole measurement take
        @param number_trajectories: how many trajectories to generate
        @param additive_noise_stddev: add normally distributed noise with the given stddev in all directions
                                        [unit: pixel]
        @param splits: how many tracks should have missing measurements in between
        @param belt_width: width of the belt (left is zero)
        @param belt_height: height of the belt (bottom is zero)

        Attention: x-coordinate is along the height of the belt.
               y-coordinate is along the width of the belt.
        """
        self.timesteps = timesteps
        self.n_dim = 2
        self.n_trajectories = number_trajectories
        self.belt_max_x = belt_height
        self.belt_height = belt_height
        self.belt_max_y = belt_width
        self.belt_width = belt_width
        self.additive_noise_stddev = additive_noise_stddev
        self.additive_target_stddev = additive_target_stddev
        self.splits = splits
        self.min_number_points_per_trajectory = min_number_points_per_trajectory

        self.track_data = self._generate_tracks()
        self.aligned_track_data = self._convert_tracks_to_aligned_tracks(self.track_data)
        self.seq2seq_data = self._convert_aligned_tracks_to_seq2seq_data(self.aligned_track_data)

    def _generate_tracks(self):
        tracks = []

        # the nan value used for every dimension of a skipped timestep
        self.nan_value_ary = [self.nan_value, self.nan_value]

        step_length = self.belt_width / self.timesteps

        # for every trajectory
        for track_number in range(self.n_trajectories):
            # pick a random point in time to start the trajectory
            start_timestep = random.randint(0, self.timesteps-self.min_number_points_per_trajectory)

            # list containing all points: add NaNs for every skipped timestep
            trajectory = [self.nan_value_ary for _ in range(start_timestep)]

            # spawn the new trajectory on the left side
            start_x = start_timestep * step_length
            start_y = random.randint(0, self.belt_max_x)
            trajectory.append([start_x, start_y])

            # end on the right side
            end_x = self.belt_width
            end_y = start_y + random.normalvariate(0, self.additive_target_stddev)

            # change in x and y direction has the maximum length of step_length
            #  because the particles move with the same velocity.
            delta_x = (end_x-start_x)
            delta_y = (end_y-start_y)
            if -1 < (delta_y / delta_x) < 1:
                alpha = np.arcsin(delta_y / delta_x)
            else:
                alpha = 0

            # corrected differences per step
            dx = np.cos(alpha) * step_length
            dy = np.sin(alpha) * step_length

            # iterate over all the time steps from start_time_step+1   until end
            for t in range(start_timestep + 2, self.timesteps + 1):
                # generate next position
                new_x = trajectory[-1][0] + dx
                new_y = trajectory[-1][1] + dy + random.normalvariate(0, self.additive_noise_stddev)
                trajectory.append([new_x, new_y])

            tracks.append(trajectory)

        return np.array(tracks)

    def _convert_tracks_to_aligned_tracks(self, track_data):
        """
        Align the tracks, in order that they all start with time step = 0

        :param track_data:
        :return:
        """
        aligned_tracks = []

        # for every track
        for track_idx in range(track_data.shape[0]):
            # for every time step
            skip_counter = 0
            track = []
            for t in range(track_data.shape[1]):
                # skip if NaN
                if np.all(track_data[track_idx, t] == self.nan_value):
                    skip_counter += 1
                    continue
                else:
                    track.append(track_data[track_idx, t])

            for t in range(skip_counter):
                track.append(self.nan_value_ary)

            aligned_tracks.append(track)

        return np.array(aligned_tracks)

    def _convert_aligned_tracks_to_seq2seq_data(self, aligned_track_data):
        seq2seq_data = []

        # for every track we create:
        #  x, y, x_target, y_target nan_value-padded
        for track_number in range(aligned_track_data.shape[0]):
            # make sure the last entry of the input arrays are nan-values
            x = aligned_track_data[track_number][:, 0]
            x = np.concatenate((x[:-1], np.array([self.nan_value])))
            y = aligned_track_data[track_number][:, 1]
            y = np.concatenate((y[:-1], np.array([self.nan_value])))

            # remove the last input because we have no target for it
            last_index = self.get_last_timestep_of_track(x) - 1
            x[last_index] = self.nan_value
            y[last_index] = self.nan_value

            # the ground truth where the particle will be
            x_target = np.concatenate((aligned_track_data[track_number][1:, 0].copy(), np.array([self.nan_value])))
            y_target = np.concatenate((aligned_track_data[track_number][1:, 1].copy(), np.array([self.nan_value])))

            input_matrix = np.vstack((x, y, x_target, y_target))

            # initialize the array with nan-value
            matrix = np.ones([self.timesteps, self.input_dim * 2]) * self.nan_value

            # insert the data of the track into the zero background (example: nan-value=0 -> black background)
            matrix[0:x.size, 0:self.input_dim * 2] = input_matrix.T

            seq2seq_data.append(matrix)

        return np.array(seq2seq_data)

    def get_track_data(self):
        return self.track_data

    def get_aligned_track_data(self):
        return self.aligned_track_data

    def get_seq2seq_data(self, nan_value=0):
        return self.seq2seq_data

    def get_mlp_data(self):
        pass

    def get_measurement_at_timestep(self, timestep):
        data = self.get_track_data()[:, timestep, :]
        return data


if __name__ == '__main__':
    f = FakeDataSet()
    f.plot_random_tracks()
