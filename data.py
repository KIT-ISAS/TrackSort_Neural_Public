# data.py
# -------
# - AbstractDataSet: Abstract class which defines the necessary interface for every data set
#    - FakeDataSet: Randomly created lines with noise
#    - CsvDataSet: Loads (real) tracks from a glob pattern of *.csv files

import io
import glob
import random
import code

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class AbstractDataSet(ABC):
    # dimensions of the belt in pixels
    belt_width = 2000
    belt_height = 2000

    # the nan_value is used for padding and must not appear in the data
    nan_value = 0
    # the number of dimensions an observation has: for example (x,y) has 2
    input_dim = 2

    timesteps = None

    batch_size = 128
    train_split_ratio = 0.5
    test_split_ratio = 0.5

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

    def get_particles(self):
        """
        Returns a list of particles.
        Each particle is a list of pairs.
        The first element of the pair is the measurement point.
        The second element is the timestep of the measurement.
        """
        track_data = self.get_aligned_track_data()
        particles = []
        for track_idx in range(track_data.shape[0]):
            is_started = False
            for time_idx in range(track_data.shape[1]):
                try:
                    if not is_started and (track_data[track_idx][time_idx] != np.array([self.nan_value, self.nan_value])).all():
                        is_started = True
                        particles.append([[time_idx, track_data[track_idx][time_idx]]])
                    elif is_started and not (track_data[track_idx][time_idx] == np.array([self.nan_value, self.nan_value])).all():
                        particles[-1].append([time_idx, track_data[track_idx][time_idx]])
                    elif is_started and (track_data[track_idx][time_idx] == np.array([self.nan_value, self.nan_value])).all():
                        break
                except Exception as exp:
                    print('error in get_particles')
                    code.interact(local=dict(globals(), **locals()))
            '''if not is_started: # TODO apparently there are empty tracks!!!
                print('something went wrong in get_particles')
                code.interact(local=dict(globals(), **locals()))'''
        return particles

    def get_measurement_at_timestep(self, timestep):
        """
        Return a numpy array with shape: [number_points, point_dimension]

        This is something like:
            >>> data = self.get_track_data()[:, timestep, :]
            >>> data[~np.isnan(data)]
        """
        raise NotImplementedError

    def plot_track(self, track, color='black', start=0, end=-1, label='track', fit_scale_to_content=False, legend=True):
        track = track[start:end]

        axes = plt.gca()
        axes.set_aspect('equal')
        if not fit_scale_to_content:
            axes.set_xlim([0, self.belt_width + 100])
            axes.set_ylim([0, self.belt_height + 100])

        plt.xlabel('x position [px]')
        plt.ylabel('y position [px]')

        axes.scatter(track[:, 0], track[:, 1], color=color)
        axes.plot(track[:, 0], track[:, 1], color=color, label=label)

        if legend:
            axes.legend()

        return axes

    def plot_random_tracks(self, n=15, end_time_step=None, fit_scale_to_content=False):
        tracks = self.get_aligned_track_data()
        for _ in range(n):
            random_index = random.randint(0, tracks.shape[0] - 1)
            random_track = tracks[random_index]
            axes = self.plot_track(random_track,
                                   color=np.random.rand(3),
                                   end=end_time_step if end_time_step else self.get_last_timestep_of_track(
                                       random_track) - 1,
                                   label='track {}'.format(random_index),
                                   fit_scale_to_content=fit_scale_to_content
                                   )
            plt.title('Tracks')
        plt.show()

    def plot_tracks_with_predictions(self, input_tracks, target_tracks, predicted_tracks, denormalize=False,
                                     display=True, tf_summary_name=None, tf_summary_step=1,
                                     start_time_step=0, end_time_step=None, fit_scale_to_content=False):
        """
        Plot the input_tracks, target_tracks and predicted_tracks into one plot.
        Set the tf_summary_name in order to save to tf summary.


        :param input_tracks:
        :param target_tracks:
        :param predicted_tracks:
        :param denormalize:
        :param display:
        :param tf_summary_name:
        :param tf_summary_step:
        :param start_time_step:
        :param end_time_step:
        :param fit_scale_to_content:
        :return:
        """
        if denormalize:
            input_tracks = self.denormalize_tracks(input_tracks)
            target_tracks = self.denormalize_tracks(target_tracks)
            predicted_tracks = self.denormalize_tracks(predicted_tracks)

        for track_idx in range(input_tracks.shape[0]):
            seq_length = self.get_last_timestep_of_track(input_tracks[track_idx])
            if end_time_step:
                seq_length = end_time_step
            axes = self.plot_track(input_tracks[track_idx], start=start_time_step,
                                   fit_scale_to_content=fit_scale_to_content,
                                   color='black', end=seq_length, label="Input truth {}".format(track_idx))
            axes = self.plot_track(target_tracks[track_idx], start=start_time_step,
                                   fit_scale_to_content=fit_scale_to_content,
                                   color='green', end=seq_length, label="Output truth {}".format(track_idx))
            axes = self.plot_track(predicted_tracks[track_idx], start=start_time_step,
                                   fit_scale_to_content=fit_scale_to_content,
                                   color='blue', end=seq_length, label="Prediction {}".format(track_idx))
            plt.title('Track with predictions')

        # store image in tf summary
        if tf_summary_name:
            # store plot in memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # add batch dim
            image = tf.expand_dims(image, 0)
            tf.summary.image(tf_summary_name, image, step=tf_summary_step)

        if display:
            plt.show()
        else:
            plt.clf()

    def plot_random_predictions(self, model, dataset, n=5, tf_summary_step=1, display=False, tf_summary_name=None,
                                denormalize=True, fit_scale_to_content=False, end_time_step=None):
        """
        Sample n random tracks from the dataset.
        Then use the model to make prediction und plot everything.
        If desired, the plots can be written to a tf summary image.

        Example:
            >>> self.plot_random_predictions(model, dataset, n=1, display=True, \
                                                      fit_scale_to_content=True, end_time_step=10)

        :param model:
        :param dataset:
        :param n: has to be smaller than batch size
        :param step:
        :param display:
        :return:
        """

        assert n <= self.batch_size

        for input_batch, target_batch in dataset.take(1):
            # reset model state
            hidden = model.reset_states()
            batch_predictions = model(input_batch)

            self.plot_tracks_with_predictions(input_batch[0:n].numpy(),
                                              target_batch[0:n].numpy(),
                                              batch_predictions[0:n].numpy(),
                                              denormalize=denormalize, display=display, tf_summary_name=tf_summary_name,
                                              tf_summary_step=tf_summary_step,
                                              fit_scale_to_content=fit_scale_to_content,
                                              end_time_step=end_time_step
                                              )

    def normalize_tracks(self, tracks, is_seq2seq_data=False):
        """
        The values within a track a normalized -> ]0, 1].
        The zero is excluded because assume that the input data also
        does not contain zeros except as padding value (nan_value).

        :param tracks:
        :param is_seq2seq_data:
        :return:
        """
        tracks[:, :, 0] /= self.belt_width
        tracks[:, :, 1] /= self.belt_width

        if is_seq2seq_data:
            tracks[:, :, 2] /= self.belt_width
            tracks[:, :, 3] /= self.belt_width
        return tracks

    def denormalize_tracks(self, tracks, is_seq2seq_data=False):
        tracks[:, :, 0] *= self.belt_width
        tracks[:, :, 1] *= self.belt_width

        if is_seq2seq_data:
            tracks[:, :, 2] *= self.belt_width
            tracks[:, :, 3] *= self.belt_width
        return tracks

    def split_train_test(self, tracks, test_ratio=0.1):
        train_tracks, test_tracks = train_test_split(tracks, test_size=test_ratio)
        return train_tracks, test_tracks

    def get_seq2seq_data_and_labels(self, normalized=True):
        tracks = self.get_seq2seq_data().copy()
        if normalized:
            tracks = self.normalize_tracks(tracks, is_seq2seq_data=True)
        input_seq = tracks[:, :, :2]
        target_seq = tracks[:, :, 2:]

        return input_seq, target_seq

    def get_tf_data_sets_seq2seq_data(self, normalized=True, test_ratio=0.1):
        tracks = self.get_seq2seq_data()
        if normalized:
            tracks = self.normalize_tracks(tracks, is_seq2seq_data=True)
        train_tracks, test_tracks = self.split_train_test(tracks, test_ratio=test_ratio)

        raw_train_dataset = tf.data.Dataset.from_tensor_slices(train_tracks)
        raw_test_dataset = tf.data.Dataset.from_tensor_slices(test_tracks)

        # for optimal shuffling the shuffle buffer has to be of the size of the number of tracks
        minibatches_train = raw_train_dataset.shuffle(train_tracks.shape[0]).batch(self.batch_size, drop_remainder=True)
        minibatches_test = raw_test_dataset.shuffle(test_tracks.shape[0]).batch(self.batch_size, drop_remainder=True)

        self.minibatches_train = minibatches_train
        self.minibatches_test = minibatches_test

        def split_input_target(chunk):
            # split the tensor (x, y, x_target, y_target)
            #  -> into two tensors (x, y) and (x_target, y_target)
            input_seq = chunk[:, :, :2]
            target_seq = chunk[:, :, 2:]
            return input_seq, target_seq

        dataset_train = minibatches_train.map(split_input_target)
        dataset_test = minibatches_test.map(split_input_target)

        return dataset_train, dataset_test

    def get_last_timestep_of_track(self, track):
        i = None
        for i in range(track.shape[0]):
            if np.all(track[i] == [self.nan_value, self.nan_value]):
                return i
        return i

    def get_longest_track(self):
        longest_track = 0

        for track_number in range(aligned_track_data.shape[0]):
            x = self.aligned_track_data[track_number][:, 0]

            last_index = self.get_last_timestep_of_track(x) - 1

            if longest_track < last_index:
                longest_track = last_index

        return longest_track

    def _convert_aligned_tracks_to_seq2seq_data(self, aligned_track_data):
        seq2seq_data = []

        longest_track = 0

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

            # find the longest track
            if longest_track < last_index:
                longest_track = last_index

            # the ground truth where the particle will be
            x_target = np.concatenate((aligned_track_data[track_number][1:, 0].copy(), np.array([self.nan_value])))
            y_target = np.concatenate((aligned_track_data[track_number][1:, 1].copy(), np.array([self.nan_value])))

            input_matrix = np.vstack((x, y, x_target, y_target))

            # initialize the array with nan-value
            matrix = np.ones([self.timesteps, self.input_dim * 2]) * self.nan_value

            # insert the data of the track into the zero background (example: nan-value=0 -> black background)
            matrix[0:x.size, 0:self.input_dim * 2] = input_matrix.T

            seq2seq_data.append(matrix)

        self.longest_track = longest_track

        print(longest_track)

        return np.array(seq2seq_data)[:, :longest_track + 1, :]

    def get_box_plot(self, model, dataset):
        maes = []

        for input_batch, target_batch in dataset:
            hidden = model.reset_states()
            batch_predictions = model(input_batch)
            for track_id in range(batch_predictions.shape[0]):
                last_id = self.get_last_timestep_of_track(input_batch[track_id])
                mae_per_timestep = np.sum(
                    ((target_batch[track_id][:last_id] - batch_predictions[track_id][:last_id]) * self.belt_width) ** 2,
                    axis=1)
                mean_mae_per_track = np.mean(np.sqrt(mae_per_timestep))
                maes.append(mean_mae_per_track)

        maes = np.array(maes)

        fig1, ax1 = plt.subplots()
        ax1.yaxis.grid(True)
        ax1.set_title('Boxplot')
        ax1.boxplot(maes, showfliers=False)


class FakeDataSet(AbstractDataSet):
    def __init__(self, timesteps=35, number_trajectories=100,
                 additive_noise_stddev=5, splits=0, additive_target_stddev=100,
                 min_number_points_per_trajectory=20, batch_size=128,
                 belt_width=2000, belt_height=2000, nan_value=0, step_length=70):
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
        self.batch_size = batch_size
        self.nan_value = nan_value
        self.step_length = step_length

        self.track_data = self._generate_tracks()
        self.aligned_track_data = self._convert_tracks_to_aligned_tracks(self.track_data)
        self.seq2seq_data = self._convert_aligned_tracks_to_seq2seq_data(self.aligned_track_data)

        # if we don't have enough tracks, then the smaller split (usually test) is so small that is smaller
        # than a batch. Because we use drop_remainder=True we cannot allow this, or else the only batch
        # would be empty -> as a result we would not have test data
        assert self.n_trajectories * min(self.test_split_ratio,
                                         self.train_split_ratio) > self.batch_size, "min(len(test_split), len(train_split)) < batch_size is not allowed! -> increase number_trajectories"

    def _generate_tracks(self):
        tracks = []

        # the nan value used for every dimension of a skipped timestep
        self.nan_value_ary = [self.nan_value, self.nan_value]

        step_length = self.step_length

        # for every trajectory
        for track_number in range(self.n_trajectories):
            # pick a random point in time to start the trajectory
            start_timestep = random.randint(0, self.timesteps - self.min_number_points_per_trajectory)

            # list containing all points: add NaNs for every skipped timestep
            trajectory = [self.nan_value_ary for _ in range(start_timestep)]

            # spawn the new trajectory on the left side
            start_x = random.randint(0, self.belt_width // 10)
            start_y = random.randint(0, self.belt_height)
            trajectory.append([start_x, start_y])

            # end on the right side
            end_x = self.belt_width
            end_y = start_y + random.normalvariate(0, self.additive_target_stddev)

            # change in x and y direction has the maximum length of step_length
            #  because the particles move with the same velocity.
            delta_x = (end_x - start_x)
            delta_y = (end_y - start_y)
            if -1 < (delta_y / delta_x) < 1:
                alpha = np.arcsin(delta_y / delta_x)
            else:
                alpha = 0

            # corrected differences per step
            dx = np.cos(alpha) * step_length
            dy = np.sin(alpha) * step_length

            track_done = False

            # iterate over all the time steps from start_time_step+1   until end
            for t in range(start_timestep + 2, self.timesteps + 1):
                # generate next position
                new_x = trajectory[-1][0] + dx
                new_y = trajectory[-1][1] + dy + random.normalvariate(0, self.additive_noise_stddev)

                # if particle is outside of the belt add nan_values
                if not track_done and (new_x > end_x or new_y > self.belt_height or new_y < 0):
                    track_done = True

                if track_done:
                    trajectory.append([self.nan_value, self.nan_value])
                else:
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

    def get_track_data(self):
        return self.track_data

    def get_aligned_track_data(self):
        return self.aligned_track_data

    def get_seq2seq_data(self, nan_value=0):
        return self.seq2seq_data

    def get_mlp_data(self):
        pass

    def get_measurement_at_timestep(self, timestep, normalized=True):
        data = self.get_track_data()[:, [timestep], :].copy()
        if normalized:
            return self.normalize_tracks(data, is_seq2seq_data=False)
        return data


class CsvDataSet(AbstractDataSet):
    def __init__(self, glob_file_pattern, min_number_detections=6, nan_value=0, input_dim=2,
                 timesteps=35, batch_size=128):
        self.glob_file_pattern = glob_file_pattern
        self.file_list = sorted(glob.glob(glob_file_pattern))
        assert len(self.file_list) > 0, "No files found"

        self.min_number_detections = min_number_detections

        self.nan_value = nan_value
        self.input_dim = input_dim
        self.batch_size = batch_size

        # ToDo(Daniel): self.num_time_steps = len(self._extract_longest_track())
        self.timesteps = timesteps
        self.tracks = self._load_tracks()

        # we assume the csv data is aligned by default
        self.aligned_tracks = self.tracks
        self.seq2seq_data = self._convert_aligned_tracks_to_seq2seq_data(self.aligned_tracks)

        # normalize in all dimensions with the same factor
        self.normalization_constant = np.nanmax(self.seq2seq_data) * 1.1  # this leaves room for tracks at the borders
        self.belt_width = self.normalization_constant

    def _load_tracks(self):
        tracks = []

        for file_ in self.file_list:
            # read the tracks from one measurement
            df = pd.read_csv(file_)

            # remove columns with less then 6 detections (same as Tobias did)
            df = df.dropna(axis=1, thresh=self.min_number_detections, inplace=False)

            # there are two columns per track, for example "TrackID_4_X" and "TrackID_4_Y"
            number_of_tracks = int((df.shape[1]) / 2)

            # We wan't to use 0.0 as NaN value. Therefore we have to check that it does not
            #   exist in the data.   Note: the double .min().min() is necessary because we
            #   first get the column minima and then we get the table minimum from that
            assert df.min().min() > 0.0, "Error: The dataframe {} contains a minimum <= 0.0".format(file_)

            # ToDo: calc track length!
            longest_track = self.timesteps  # df.count().max()

            # for every track we create:
            # - track
            for track_number in range(number_of_tracks):
                # create the simple tracks
                # **Attention:** the columns are ordered as (Y,X) and we turn it around to (X,Y)
                track = df.iloc[:, [(2 * track_number + 1), (2 * track_number)]].to_numpy(copy=True)
                background = np.zeros([longest_track, self.input_dim])
                background[:track.shape[0], :track.shape[1]] = track
                tracks.append(np.nan_to_num(background))

        tracks = np.array(tracks)
        return tracks

    def get_track_data(self):
        return self.tracks

    def get_aligned_track_data(self):
        return self.aligned_tracks

    def get_seq2seq_data(self, nan_value=0):
        return self.seq2seq_data

    def get_mlp_data(self):
        pass

    def get_measurement_at_timestep(self, timestep, normalized=True):
        data = self.get_track_data()[:, [timestep], :].copy()
        if normalized:
            return self.normalize_tracks(data, is_seq2seq_data=False)
        return data


if __name__ == '__main__':
    f = FakeDataSet()
    f.plot_random_tracks()
