# data.py
# -------
# - AbstractDataSet: Abstract class which defines the necessary interface for every data set
#    - FakeDataSet: Randomly created lines with noise
#    - CsvDataSet: Loads (real) tracks from a glob pattern of *.csv files

import io
import glob
import random
import code
import logging

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class AbstractDataSet(ABC):
    # dimensions of the belt in pixels
    # ToDo: Currently this has to be the same value
    #       assert belt_width == belt_height
    belt_width = 2000
    belt_height = 2000
    normalization_constant = 2000

    # the nan_value is used for padding and must not appear in the data
    nan_value = 0
    # the number of dimensions an observation has: for example (x,y) has 2
    input_dim = 2

    timesteps = None
    longest_track = None

    train_split_ratio = 0.5
    test_split_ratio = 0.5

    seq2seq_data = np.array([])
    track_data = np.array([])
    aligned_track_data = np.array([])
    mlp_data = np.array([])
    track_num_mlp_id = dict()
    separation_track_data = np.array([])
    separation_spatial_labels = np.array([])
    separation_temporal_labels = np.array([])

    def get_nan_value_ary(self):
        return [self.nan_value, self.nan_value]
        
    def get_seq2seq_data(self, nan_value=0):
        """
        Return a numpy array which can be used for RNN training.
        All trajectories start at timestep 0 and end at timestep max_timestep.
        max_timestep is the length of the longest trajectory. All other
        trajectories are padded with nan_value.

        shape: [number_tracks, max_timestep, point_dimension * 2]
        """
        return self.seq2seq_data

    def get_mlp_data(self):
        """
        Return a numpy array which can be used for MLP Training.
        The input are n datapoints and the label is the next datapoint which
        should be predicted.
        Example: [ [[0,0], [0,1], [0,2], [0,3]],   [0,4] ]
        TODO implement.
        """
        pass

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
        return self.track_data

    def get_num_timesteps(self):
        """The number of time steps of the full data set.
        This is not dimensions used for the RNN unrolling.
        """
        return self.timesteps

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
        return self.aligned_track_data

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
                    if not is_started and (
                            track_data[track_idx][time_idx] != np.array([self.nan_value, self.nan_value])).all():
                        is_started = True
                        particles.append([[time_idx, track_data[track_idx][
                            time_idx] / self.normalization_constant]])
                    elif is_started and not (
                            track_data[track_idx][time_idx] == np.array([self.nan_value, self.nan_value])).all():
                        particles[-1].append([time_idx, track_data[track_idx][
                            time_idx] / self.normalization_constant])
                    elif is_started and (
                            track_data[track_idx][time_idx] == np.array([self.nan_value, self.nan_value])).all():
                        break
                except Exception as exp:
                    logging.error('error in get_particles')
                    code.interact(local=dict(globals(), **locals()))
        return particles

    def get_measurement_at_timestep(self, timestep):
        """
        Return a numpy array with shape: [number_points, point_dimension]

        This is something like:
            >>> data = self.get_track_data()[:, timestep, :]
            >>> data[~np.isnan(data)]
        """
        raise NotImplementedError

    def get_particle_timestep_data(self):
        """Create data for multitarget tracking.

        Returns a list of particles in each timestep.
            particle_time_list: list of np arrays. 
            particle_time_list[0] = np array with a entry per particle in this timestep:
                                    particle_id, x, y
        """
        raise NotImplementedError

    def get_belt_limits(self):
        """Get the belt limits based on min and max values of measurement.

        Returns: 
            belt_limits (np.array):  Limits of the belt [[x_min, x_max],[y_min, y_max]]
        """
        belt_limits = np.zeros([2,2])
        masked_tracks = np.ma.array(self.aligned_track_data, mask=self.aligned_track_data==0)
        belt_limits[0,0] = np.min(masked_tracks[:,:,0])
        belt_limits[0,1] = np.max(masked_tracks[:,:,0])
        belt_limits[1,0] = np.min(masked_tracks[:,:,1])
        belt_limits[1,1] = np.max(masked_tracks[:,:,1])
        return belt_limits/self.normalization_constant

    def get_measurement_at_timestep_list(self, timestep, normalized=True):
        data = self.get_measurement_at_timestep(timestep)
        data = list(map(lambda x: np.squeeze(data[x]), filter(lambda x: data[x][0][0] > 0, range(data.shape[0]))))
        return data

    def plot_track(self, track, color='black', start=0, end=-1, label='track', fit_scale_to_content=False, legend=True):
        track = track[start:end]

        axes = plt.gca()
        axes.set_aspect('equal')
        if not fit_scale_to_content:
            axes.set_xlim([0, self.belt_width * 1.1])
            axes.set_ylim([0, self.belt_height * 1.1])

        plt.xlabel('x position')
        plt.ylabel('y position')

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

        #assert n <= self.batch_size

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
        tracks[:, :, :2] /= self.normalization_constant

        if is_seq2seq_data:
            tracks[:, :, 2] /= self.normalization_constant
            tracks[:, :, 3] /= self.normalization_constant
        return tracks

    def normalize_data(self, data):
        """Normalize a set of values with self.normalization_constant"""
        return data / self.normalization_constant

    def denormalize_tracks(self, tracks, is_seq2seq_data=False):
        tracks[:, :, :2] *= self.normalization_constant

        if is_seq2seq_data:
            tracks[:, :, 2] *= self.normalization_constant
            tracks[:, :, 3] *= self.normalization_constant
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

    def get_tf_data_sets_seq2seq_data(self, normalized=True, test_ratio=0.1, batch_size=64, random_seed = None):
        tracks = self.get_seq2seq_data()
        if normalized:
            tracks = self.normalize_tracks(tracks, is_seq2seq_data=True)
        
        # Create ids for train and test tracks with optional random seed
        train_ids, test_ids = train_test_split(np.arange(tracks.shape[0]), test_size=test_ratio, random_state=random_seed)
        
        train_tracks = tracks[train_ids]
        test_tracks = tracks[test_ids]

        raw_train_dataset = tf.data.Dataset.from_tensor_slices(train_tracks)
        raw_test_dataset = tf.data.Dataset.from_tensor_slices(test_tracks)

        # for optimal shuffling the shuffle buffer has to be of the size of the number of tracks
        if random_seed is None:
            minibatches_train = raw_train_dataset.shuffle(train_tracks.shape[0]).batch(batch_size, drop_remainder=True)
            minibatches_test = raw_test_dataset.shuffle(test_tracks.shape[0]).batch(batch_size, drop_remainder=True)
            logging.warning("Always use a random seed if you want to combine MLPs and RNNs/KFs!")
        else:
            minibatches_train = raw_train_dataset.batch(batch_size, drop_remainder=True)
            minibatches_test = raw_test_dataset.batch(batch_size, drop_remainder=True)

        

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

    def get_tf_data_sets_mlp_data(self, normalized=True, test_ratio=0.1, batch_size=64, random_seed = None):
        """Seperate MLP data in training and test set and convert them to Tensor data.

        Set random_seed to not None to fix the shuffling of training and test data.

        Args:
            normalized (Boolean):   Normalize the data
            test_ratio (double):    Rate of data that is turned into test data
            batch_size (int):       Batch size for training and test data
            random_seed (int):      None: Random shuffling; int: Fixed shuffling

        Returns:
            dataset_train, dataset_test (tf.Tensor): Train and test data in tensorflow format
        """
        tracks = self.aligned_track_data
        mlp_data = self.mlp_data
        if normalized:
            mlp_data = self.normalize_data(mlp_data)
        
        # Create ids for train and test tracks with optional random seed
        train_track_ids, test_track_ids = train_test_split(np.arange(tracks.shape[0]), test_size=test_ratio, random_state=random_seed)
        
        # Get the mlp data ids of the track ids
        train_ids = []
        test_ids = []
        for track_id in train_track_ids:
            train_ids.extend(self.track_num_mlp_id.get(track_id))
        for track_id in test_track_ids:
            test_ids.extend(self.track_num_mlp_id.get(track_id))
        
        train_data = mlp_data[train_ids]
        test_data = mlp_data[test_ids]

        raw_train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        raw_test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

        # for optimal shuffling the shuffle buffer has to be of the size of the number of tracks
        if random_seed is None:
            minibatches_train = raw_train_dataset.shuffle(train_data.shape[0]).batch(batch_size * self.longest_track, drop_remainder=True)
            minibatches_test = raw_test_dataset.shuffle(test_data.shape[0]).batch(batch_size * self.longest_track, drop_remainder=True)
            logging.warning("Always use a random seed if you want to combine MLPs and RNNs/KFs!")
        else:
            minibatches_train = raw_train_dataset.batch(batch_size * self.longest_track, drop_remainder=True)
            minibatches_test = raw_test_dataset.batch(batch_size * self.longest_track, drop_remainder=True)

        def split_input_target(chunk):
            # split the tensor (x, y, x_target, y_target)
            #  -> into two tensors (x, y) and (x_target, y_target)
            input_seq = chunk[:, :-2]
            target_seq = chunk[:, -2:]
            return input_seq, target_seq

        dataset_train = minibatches_train.map(split_input_target)
        dataset_test = minibatches_test.map(split_input_target)

        return dataset_train, dataset_test

    def mlp_target_to_track_format(self, mlp_target):
        """Convert a batch of MLP targets/predictions into tracks

        Args:
            mlp_target (np.array):  The target array to convert

        Returns:
            track_target (np array): The target in track format
        """ 
        assert(mlp_target.shape[0] % self.longest_track == 0)

        n_tracks = int(mlp_target.shape[0]/self.longest_track)
        track_target = np.reshape(mlp_target, [n_tracks, self.longest_track, 2])
        return track_target


    def get_tf_data_sets_seq2seq_with_separation_data(self, normalized=True, 
                                                      test_ratio=0.1,
                                                      batch_size=64, 
                                                      random_seed=None,
                                                      time_normalization=22.,
                                                      only_last_timestep_additional_loss = True):
        """Get the training and test datasets for RNN separation prediction.

        input format: [[x_0, y_0], ..., [x_n, y_n]]
        target format: [[x_1, y_1, y_nozzle, dt_nozzle]]
            dt_nozzle describes the difference in time from the last measurement to the nozzle array.

        Produce two masks for the separation prediction:
            tracking_mask: This mask is True for all (inp, target) pairs along the track
            separation_mask: This mask is only True in the last valid (inp, target) pair in the track if only_last_timestep_additional_loss == True.
                             This can be used to only take the last prediction of the RNN into account in the separation prediction.
                             If only_last_timestep_additional_loss==False, the separation mask equals the tracking mask.

        Args:
            normalized (Boolean):   Normalize the data
            test_ration (double):   Ratio of test data to all data. Value between 0 and 1.
            batch_size (int):       Batch size for training. This value gets multiplied with the total track length to generate more reasonable batch sizes that match the RNN data.
            random_seed (int):      Random seed for train/test split. Set this to None to be random every time.
            time_normalization (double):  Normalize the time step target with this parameter
            only_last_timestep_additional_loss (Boolean): Sets the style of the separation mask.

        Returns:
            training_dataset (tf.Dataset): Batches of (input, target, tracking_mask, separation_mask) pairs for training
            testing_dataset (tf.Dataset): Batches of (input, target, tracking_mask, separation_mask) pairs for testing
        """
        track_data = self.separation_track_data
        spatial_labels = self.separation_spatial_labels
        temporal_labels = self.separation_temporal_labels
        tracks = self._convert_aligned_tracks_to_seq2seq_data(track_data)
        num_time_steps = tracks.shape[1]

        # add additional labels
        # track[0,0,:] = [x_1, y_1, x_2, y_2, y_nozzle, t_nozzle]
        tracks = np.concatenate(
            (tracks,
             np.repeat(spatial_labels[:, 1][None], num_time_steps, axis=0).T.reshape(-1, tracks.shape[1], 1),
             np.repeat(temporal_labels[None], num_time_steps, axis=0).T.reshape(-1, tracks.shape[1], 1)
             ),
            axis=-1
        )
        # Build tracking and separation mask
        tracking_mask = np.all(tracks[:,:,:4]!=self.nan_value, axis=2)
        if only_last_timestep_additional_loss:
            mask_length = np.sum(tracking_mask, axis=1)
            separation_mask = np.zeros(tracking_mask.shape, dtype=bool)
            separation_mask[np.arange(mask_length.size), mask_length] = True
        else:
            separation_mask = tracking_mask
        # Fill spatial and temporal labels with 0 in areas with no measurement
        #tracks[~tracking_mask,4:6] = self.nan_value
        if normalized:
            tracks[:, :, [0, 1, 2, 3]] /= self.normalization_constant
            # normalize spatial prediction
            tracks[:, :, [4]] /= self.normalization_constant
            # normalize temporal prediction
            tracks[:, :, [5]] /= time_normalization

        tracks = np.concatenate((tracks, tracking_mask[:,:,np.newaxis], separation_mask[:,:,np.newaxis]), axis=-1)

        train_tracks, test_tracks = train_test_split(tracks, test_size=test_ratio, random_state=random_seed)

        raw_train_dataset = tf.data.Dataset.from_tensor_slices(train_tracks)
        raw_test_dataset = tf.data.Dataset.from_tensor_slices(test_tracks)

        # for optimal shuffling the shuffle buffer has to be of the size of the number of tracks
        if random_seed is None:
            minibatches_train = raw_train_dataset.shuffle(train_tracks.shape[0]).batch(batch_size=batch_size, drop_remainder=True)
            minibatches_test = raw_test_dataset.shuffle(test_tracks.shape[0]).batch(batch_size=batch_size, drop_remainder=True)
        else:
            minibatches_train = raw_train_dataset.batch(batch_size=batch_size, drop_remainder=True)
            minibatches_test = raw_test_dataset.batch(batch_size=batch_size, drop_remainder=True)

        def split_input_target_separation(chunk):
            # split the tensor (x, y, y_pred, t_pred, x_target, y_target, y_pred_target, t_pred_target)
            #  -> into two tensors (x, y) and (x_target, y_target)
            input_seq = chunk[:, :, :2]
            target_seq = chunk[:, :, 2:6]
            tracking_mask = chunk[:, :, 6]
            separation_mask = chunk[:, :, 7]
            return input_seq, target_seq, tracking_mask, separation_mask

        dataset_train = minibatches_train.map(split_input_target_separation)
        dataset_test = minibatches_test.map(split_input_target_separation)

        return dataset_train, dataset_test, num_time_steps

    def get_tf_data_sets_mlp_with_separation_data(self, 
                                                  normalized=True, 
                                                  test_ratio=0.1,
                                                  batch_size=64, 
                                                  random_seed=None,
                                                  time_normalization=22.,
                                                  n_inp_points = 5):
        """Get the training and test datasets for MLP separation prediction.

        input format: [x_1, x_2, ..., x_n_inp, y_1, y_2, ..., y_n_inp]
        target format: [y_nozzle, dt_nozzle]
            dt_nozzle describes the difference in time from the last measurement to the nozzle array.
            We don't take absolute time values because this minimizes the input to the network.
        mask format: [is_masked]: Tracks with less than n_inp_points should be masked

        If there are not enough measurements in the track, the default input/target is: [0, 0, ..., 0], [0, 0]

        Args:
            normalized (Boolean):   Normalize the data
            test_ration (double):   Ratio of test data to all data. Value between 0 and 1.
            batch_size (int):       Batch size for training. This value gets multiplied with the total track length to generate more reasonable batch sizes that match the RNN data.
            random_seed (int):      Random seed for train/test split. Set this to None to be random every time.
            time_normalization (double):  Normalize the time step target with this parameter
            n_inp_points (int):     Number of measurements for the input of the MLP

        Returns:
            training_dataset (tf.Dataset): Batches of (input, target, mask) sets for training
            testing_dataset (tf.Dataset): Batches of (input, target, mask) sets for testing
        """
        track_data = self.separation_track_data
        spatial_labels = self.separation_spatial_labels
        temporal_labels = self.separation_temporal_labels
        
        n_tracks = track_data.shape[0]
        mlp_data = np.zeros([n_tracks, 2*n_inp_points + 3])
        # Build MLP input and target for all tracks
        missing_track_counter = 0
        for i in range(n_tracks):
            n_measurements = self.get_last_timestep_of_track(track_data[i])
            if n_measurements >= n_inp_points:
                # Set input = [x1, x2, ..., y_1, y2, ...]
                c=0
                for index in np.arange(n_measurements-n_inp_points, n_measurements):
                    mlp_data[i,c] = track_data[i, index, 0]
                    mlp_data[i,c+n_inp_points] = track_data[i, index, 1]
                    c += 1
                # Set target [y_nozzle, dt_nozzle]
                mlp_data[i, -3] = spatial_labels[i, 1]
                mlp_data[i, -2] = temporal_labels[i]
                mlp_data[i, -1] = 1
            else:
                missing_track_counter += 1
                mlp_data[i, -1] = 0

        logging.info("Skipping {} tracks in the MLP separation prediction because their length was < {}".format(missing_track_counter, n_inp_points))

        if normalized:
            # normalize spatial data
            mlp_data[:, :-2] /= self.normalization_constant
            # normalize temporal prediction
            mlp_data[:, -2] /= time_normalization

        train_data, test_data = train_test_split(mlp_data, test_size=test_ratio, random_state=random_seed)

        raw_train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        raw_test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

        # for optimal shuffling the shuffle buffer has to be of the size of the number of tracks
        if random_seed is None:
            minibatches_train = raw_train_dataset.shuffle(n_tracks).batch(batch_size, drop_remainder=True)
            minibatches_test = raw_test_dataset.shuffle(n_tracks).batch(batch_size, drop_remainder=True)
        else:
            minibatches_train = raw_train_dataset.batch(batch_size, drop_remainder=True)
            minibatches_test = raw_test_dataset.batch(batch_size, drop_remainder=True)

        def split_input_target_separation(chunk):
            # split the tensor (x1, x2, ..., y1, y2, ..., y_pred_target, t_pred_target)
            #  -> into two tensors (x..., y...) and (y_pred_target, t_pred_target)
            input_seq = chunk[:, :-3]
            target_seq = chunk[:, -3:-1]
            mask_seq = chunk[:, -1]
            return input_seq, target_seq, mask_seq

        dataset_train = minibatches_train.map(split_input_target_separation)
        dataset_test = minibatches_test.map(split_input_target_separation)

        return dataset_train, dataset_test

    def get_last_timestep_of_track(self, track, use_nan=False, beginning=None):
        if beginning is None:
            beginning = self.get_first_timestep_of_track(track, use_nan=use_nan)

        if use_nan:
            for i in range(beginning, track.shape[0]):
                if np.all(np.isnan(track[i])):
                    return i
            return i+1
        else:
            for i in range(beginning, track.shape[0]):
                if np.all(track[i] == [self.nan_value, self.nan_value]):
                    return i
            return i+1

    def get_first_timestep_of_track(self, track, use_nan=False):
        if use_nan:
            for i in range(track.shape[0]):
                if not np.all(np.isnan(track[i])):
                    return i
            return i
        else:
            for i in range(track.shape[0]):
                if not np.all(track[i] == [self.nan_value, self.nan_value]):
                    return i
            return i

    def get_longest_track(self):
        longest_track = 0
        aligned_track_data = self.get_aligned_track_data()

        for track_number in range(aligned_track_data.shape[0]):
            x = aligned_track_data[track_number][:, 0]

            last_index = self.get_last_timestep_of_track(x) - 1

            if longest_track < last_index:
                longest_track = last_index

        return longest_track

    def _convert_aligned_tracks_to_seq2seq_data(self, aligned_track_data):
        """Create input to target data from aligned tracks.

        The output format will be:
        x_input, y_input, x_output, y_output

        Args:
            aligned_track_data (np.array): The tracks in format: [n_tracks, length_tracks, 2]

        Returns:
            np.array of sequence to sequence data (seq2seq), shape: [n_tracks, length_tracks, 4]
        """
        assert self.longest_track is not None, "self.longest_track not set"
        logging.info("longest_track={}".format(self.longest_track))
        n_tracks = aligned_track_data.shape[0]
        track_length = aligned_track_data.shape[1]
        seq2seq_data = np.full([n_tracks, track_length, 4], self.nan_value)
        # Fill in start positions
        seq2seq_data[:,:,:2] = aligned_track_data[:, 0:track_length]
        # Fill in end positions
        seq2seq_data[:,:-1,2:] = aligned_track_data[:, 1:track_length]
        # Delete the point where there is only a start but not and end position
        # ---> We don't do this anymore so there is the last measurement available for the separation prediction.
        #correct_pos = np.bitwise_and(seq2seq_data[:,:,0] != self.nan_value, seq2seq_data[:,:,3] == self.nan_value)
        #seq2seq_data[correct_pos,:2] = self.nan_value
        # Return it :)
        return np.array(seq2seq_data)

    def _convert_aligned_tracks_to_mlp_data(self, aligned_track_data, n_inp_points = 5):
        """Create input to target data for MLP from aligned tracks.

        The MLP data format is:
        x_input1, x_input2, ..., x_inputN, y_input1, y_input2, ...,y_inputN, x_output, y_output

        Create a array that matches each data instance to a track number
        Create a dict that matches each track all its data instances

        Args:
            aligned_track_data (np.array): The tracks in format: [n_tracks, length_tracks, 2]
            n_inp_points (int): Number of track points as input

        Returns:
            mlp_data (np.array):        The tracks in MLP data format
            mlp_id_track_num (np.array): Matches each data instance to a track number
            track_num_mlp_id (dict):    Matches each track to all its data instances
        """
        assert self.longest_track is not None, "self.longest_track not set"

        n_tracks = aligned_track_data.shape[0]
        
        mlp_data = np.full([n_tracks*self.longest_track, 2 * (n_inp_points + 1)], self.nan_value)
        mlp_id_track_num = np.zeros([n_tracks*self.longest_track])
        track_num_mlp_id = dict()
        counter = 0
        # For each track
        for track_number in range(n_tracks):
            track_num_mlp_id[track_number] = []
            track = aligned_track_data[track_number]
            # Last index with measurement
            last_index = self.get_last_timestep_of_track(track) - 1
            for i in range(self.longest_track):
                # If there are enough values to build a vector
                if i >= n_inp_points-1 and i < last_index:
                    # x_input1, x_input2, ..., x_inputN, y_input1, y_input2, ...,y_inputN
                    input_array = np.append(aligned_track_data[track_number, i-n_inp_points+1:i+1, 0], aligned_track_data[track_number, i+1-n_inp_points:i+1, 1])
                    # x_output, y_output
                    output_array = np.append(aligned_track_data[track_number, i+1, 0], aligned_track_data[track_number, i+1, 1])
                    full_array = np.append(input_array, output_array)
                    mlp_data[counter] = full_array

                # Add track id and MLP dataset id to list and dict
                mlp_id_track_num[counter]=track_number
                track_num_mlp_id[track_number].append(counter)
                counter += 1

        return mlp_data, mlp_id_track_num, track_num_mlp_id

    def get_box_plot(self, model, dataset):
        maes = []

        for input_batch, target_batch in dataset:
            hidden = model.reset_states()
            batch_predictions = model(input_batch)
            for track_id in range(batch_predictions.shape[0]):
                last_id = self.get_last_timestep_of_track(input_batch[track_id])
                mae_per_timestep = np.sum(
                    ((target_batch[track_id][:last_id] - batch_predictions[track_id][
                                                         :last_id]) * self.normalization_constant) ** 2,
                    axis=1)
                mean_mae_per_track = np.mean(np.sqrt(mae_per_timestep))
                maes.append(mean_mae_per_track)

        maes = np.array(maes)

        fig1, ax1 = plt.subplots()
        ax1.yaxis.grid(True)
        ax1.set_title('Boxplot')
        ax1.boxplot(maes, showfliers=False)

    def get_separation_prediction_data(self, virtual_belt_edge_x_position=1200, virtual_nozzle_array_x_position=1400,
                                       min_measurements_count=3):
        """
        Get training data for the separation task.
        For every track, which has measurements left of the virtual_belt_edge_x_position (visible measurements) and
        crosses the virtual_nozzle_array_x_position, return the visible measurements as data and the intersection
        as prediction label.

        The most crucial step for the belt sorting machine is the activation of the correct nozzle at the end of belt.
        In reality, the particles fall of the edge of the belt and then they have a small flight phase.
        The particles fly over the nozzle array where they are being separated.
        **Problem:** How to get training data of this process?

        1) Use an accurate physical model
        2) Assume that the flight phase is similar to the belt traveling phase

        This method uses solution 2). Therefore we:
        - use the last observation of tracks as the position where the particle crosses the nozzle array.
        - In a real setup, there is a "blind" gap between the edge of the belt and the nozzle. That is the reason why
          we don't use all measurements after a virtual belt edge.

        Graphically speaking: We add a virtual edge to the belt, which makes it shorter. Then comes the flight phase
          (where our particles still travel on the belt but we don't use its observations). Finally the particle reaches
          the virtual nozzle array, which we use as ground truth.

        Visualization:
            A) The original belt with some particles (denoted a P)

            0==============================================0               ||
            |                 P                P           |               ||
            |   P                                          |               ||
            |                   P     P                  P |               ||
            |         P                           P        |               ||
            |                                              |               ||
            0==============================================0               ||
            Begin                                        Edge         Nozzle array

            B) For the particle that leaves the belt over the edge,
               we have to predict the correct position and time
               when it will cross the nozzle.

            0==============================================0               ||
            |                 P                P           |               ||
            |   P                                          |               ||
            |                   P     P                  P----------------->|
            |         P                           P        |               ||
            |                                              |               ||
            0==============================================0               ||
            Begin                                        Edge         Nozzle array

            C) Due to the fact, that we only have training data
               when the particles are on the belt and not when they
               are in their flight phase, neither above the nozzle array,
               we introduce the virtual belt edge and virtual nozzle array.

            0==============================================0               ||
            |                 P     |           P     ||   |               ||
            |   P                   |                 ||   |               ||
            |                   P   |  P              || P |               ||
            |         P             |              P  ||   |               ||
            |                       |                 ||   |               ||
            0==============================================0               ||
            Begin             Virtual Edge     Virtual nozzles

            D) Measurements between the virtual edge and the virtual
               nozzle array are hidden ("blind area"). This is according
               to the flight area.

            0==============================================0               ||
            |                 P     |                 ||   |               ||
            |   P                   |                 ||   |               ||
            |                   P   |                 || P |               ||
            |         P             |                 ||   |               ||
            |                       |                 ||   |               ||
            0==============================================0               ||
            Begin             Virtual Edge     Virtual nozzles

            E) Label generation: The ground truth for the separation
               prediction works as follows:
               1. Find the two measurements of a track, whose connecting
                  line intersects with the virtual nozzle (in the illustration below "Q" and "P")
               2. Use the intersection as spatial prediction ground truth ("X")
               3. In order to get the temporal ground truth ("when does the
                  particle cross the array?"), we calculate the distance of
                  the intersection to the measurement left and right of it.
                  We use these distances to get a weighted average between
                  the timestamps of the two measurements.

            0==============================================0               ||
            |                       |                 ||   |               ||
            |                       |                 ||   |               ||
            |                       |               Q--X--P|               ||
            |                       |                 ||   |               ||
            |                       |                 ||   |               ||
            0==============================================0               ||
            Begin             Virtual Edge     Virtual nozzles


        :param virtual_nozzle_array_x_position:
        :param virtual_belt_edge_x_position:
        """
        assert virtual_nozzle_array_x_position < self.belt_width, "virtual_nozzle_array_x_position is too far right"
        assert virtual_belt_edge_x_position < virtual_nozzle_array_x_position, \
            "assert: virtual_belt_edge_x_position < virtual_nozzle_array_x_position"
        assert min_measurements_count > 0, "min_measurements_count has to be >0"

        aligned_track_data = self.get_aligned_track_data()

        # 1. only work with tracks which have visible measurements left of the virtual_belt_edge_x_position
        # TODO:   -> minimum of min_measurements_count necessary

        left_of_edge_mask = np.bitwise_and(aligned_track_data[:, min_measurements_count-1, 0] < virtual_belt_edge_x_position, aligned_track_data[:, min_measurements_count-1, 0] != self.nan_value)
        # apply mask as filter
        aligned_track_data = aligned_track_data[left_of_edge_mask]

        # 2. remove tracks without measurements after the virtual_nozzle_array_x_position

        # get nonzero mask
        not_null_mask = (np.sum((aligned_track_data != [self.nan_value, self.nan_value]).astype(np.int), axis=2) == 0)
        # get the last measurement index
        last_measurement_index = np.argmax(not_null_mask, axis=1) - 1
        # last measurements
        last_measurements = aligned_track_data[
            np.arange(aligned_track_data.shape[0])[:, None],
            last_measurement_index[:, None]]
        # shape: [tracks, 2]
        last_measurements = last_measurements.reshape(last_measurements.shape[0], 2)
        # filter by last measurement (is it right of the array nozzle?)
        aligned_track_data = aligned_track_data[(last_measurements[:, 0] > virtual_nozzle_array_x_position)]
        assert aligned_track_data.shape[0] > 0, "No tracks lie in the visible area"

        # 3. find the points Q and P for every track
        # Q
        distance_to_nozzle = (aligned_track_data - virtual_nozzle_array_x_position)
        distance_to_nozzle[distance_to_nozzle[:, :, 0] > 0] = -np.inf
        q_indices = distance_to_nozzle[:, :, 0].argmax(axis=1)
        q_values = aligned_track_data[
            np.arange(aligned_track_data.shape[0])[:, None],
            q_indices[:, None]
        ].reshape(-1, 2)
        # P
        distance_to_nozzle = (aligned_track_data - virtual_nozzle_array_x_position)
        distance_to_nozzle[distance_to_nozzle[:, :, 0] <= 0] = +np.inf
        p_indices = distance_to_nozzle[:, :, 0].argmin(axis=1)
        p_values = aligned_track_data[
            np.arange(aligned_track_data.shape[0])[:, None],
            p_indices[:, None]
        ].reshape(-1, 2)

        # 4. Intersect the nozzle_array with the line QP -> intersection "X"
        delta_xy = p_values - q_values
        slope = delta_xy[:, 1] / delta_xy[:, 0]
        # spatial intersection
        y_at_nozzle = q_values[:, 1] + slope * (virtual_nozzle_array_x_position - q_values[:, 0])
        spatial_labels = np.vstack((np.ones(y_at_nozzle.shape) * virtual_nozzle_array_x_position, y_at_nozzle)).T
        
        # 5. restrict dataset to measurements left of the virtual edge
        distance_to_edge = (aligned_track_data - virtual_belt_edge_x_position)
        distance_to_edge[distance_to_edge[:, :, 0] > 0] = -np.inf
        last_measurement_before_edge_indices = distance_to_edge[:, :, 0].argmax(axis=1)
        # build a mask for all tracks, which data points to keep
        mask = None
        # IMPROVEMENT: for-loop might be replaceable
        for idx in last_measurement_before_edge_indices:
            m = np.ones(aligned_track_data.shape[1])
            m[idx + 1:] = 0
            if mask is None:
                mask = m[:, None]
            else:
                mask = np.hstack((mask, m[:, None]))
        mask = mask.T.astype(np.bool)
        # hide data
        aligned_track_data[~mask, :] = self.nan_value
        # Shorten tracks to new max length
        longest_track = np.max(np.sum(np.all(aligned_track_data != self.nan_value, axis=-1), axis=-1))
        aligned_track_data = aligned_track_data[:,:longest_track]

        # 6. temporal intersection: 
        #   use the euclidean distance to calculate a weighted average between the indices of Q and P
        #   We use the distance from the last measurement to the nozzle array as time label.
        qx = np.sqrt(np.sum((spatial_labels - q_values) ** 2, axis=1))
        xp = np.sqrt(np.sum((spatial_labels - p_values) ** 2, axis=1))
        # take the weighted average between both indices
        temporal_labels = (qx * q_indices + xp * p_indices) / (qx + xp) - last_measurement_before_edge_indices
        
        return aligned_track_data, spatial_labels, temporal_labels

    def _convert_tracks_to_aligned_tracks(self, track_data):
        """
        Align the tracks, in order that they all start with time step = 0

        Args:
            track_data (np.array): Tracks that start at any time step. Nan value is 0.0, Shape: [n_tracks, n_timesteps, 2]
        
        Returns:
            aligned_tracks (np.array): All tracks start at t=0. Shape: [n_tracks, longest_track, 2]
        """
        n_tracks = track_data.shape[0]
        aligned_tracks = np.zeros([n_tracks, self.longest_track, 2])
        # for every track
        for i in range(n_tracks):
            short_track = track_data[i, (track_data[i] != (0,0)).all(axis=1)]
            aligned_tracks[i, 0:short_track.shape[0]] = short_track 

        return aligned_tracks


class FakeDataSet(AbstractDataSet):
    def __init__(self, mlp_input_dim=5, timesteps=350, number_trajectories=512,
                 additive_noise_stddev=0, splits=0, additive_target_stddev=0,
                 min_number_points_per_trajectory=20, batch_size=64,
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
        self.nan_value = nan_value
        self.step_length = step_length

        self.track_data = self._generate_tracks()
        self.aligned_track_data = self._convert_tracks_to_aligned_tracks(self.track_data)
        self.seq2seq_data = self._convert_aligned_tracks_to_seq2seq_data(self.aligned_track_data)
        #self._create_mlp_data(self.aligned_track_data, n_inp_points = mlp_input_dim)
        # if we don't have enough tracks, then the smaller split (usually test) is so small that is smaller
        # than a batch. Because we use drop_remainder=True we cannot allow this, or else the only batch
        # would be empty -> as a result we would not have test data
        assert self.n_trajectories * min(self.test_split_ratio,
                                         self.train_split_ratio) > batch_size, \
            "min(len(test_split), len(train_split)) < batch_size is not allowed! -> increase number_trajectories"

    def _generate_tracks(self):
        tracks = []

        step_length = self.step_length

        nan_value_ary = self.get_nan_value_ary()

        longest_track = 0

        # for every trajectory
        for track_number in range(self.n_trajectories):
            # pick a random point in time to start the trajectory
            start_timestep = random.randint(0, self.timesteps - self.min_number_points_per_trajectory)

            # list containing all points: add NaNs for every skipped timestep
            trajectory = [nan_value_ary for _ in range(start_timestep)]

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

            track_length = 1

            # iterate over all the time steps from start_time_step+2   until end
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
                    track_length += 1

            if track_length > longest_track:
                longest_track = track_length

            tracks.append(trajectory)

        self.longest_track = longest_track

        return np.array(tracks)

    def get_measurement_at_timestep(self, timestep, normalized=True):
        data = self.get_track_data()[:, [timestep], :].copy()
        if normalized:
            return self.normalize_tracks(data, is_seq2seq_data=False)
        return data


class CsvDataSet(AbstractDataSet):
    def __init__(self, glob_file_pattern=None, nan_value=0, input_dim=2, mlp_input_dim=5,
                 timesteps=35, data_is_aligned=True,
                 rotate_columns=False, normalization_constant=None,
                 virtual_belt_edge_x_position=1200,
                 virtual_nozzle_array_x_position=1400, 
                 min_measurements_count=3,
                 birth_rate_mean=6, birth_rate_std=2,
                 additive_noise_stddev=0,
                 is_separation_prediction=False):
        self.glob_file_pattern = glob_file_pattern
        self.file_list = sorted(glob.glob(glob_file_pattern))
        assert len(self.file_list) > 0, "No files found"
        self.rotate_columns = rotate_columns

        self.nan_value = nan_value
        self.input_dim = input_dim

        # Set timesteps if wanted. Else: _load_tracks calculates the longest track length
        if timesteps is not None:
            self.timesteps = timesteps
        self.track_data = self._load_tracks(data_is_aligned=data_is_aligned, rotate_columns=rotate_columns, min_measurements_count=min_measurements_count)

        # Add normally distributed noise to the tracks
        self.additive_noise_stddev = additive_noise_stddev
        if self.additive_noise_stddev > 0.0:
            logging.info("Add normal noise with std={}".format(self.additive_noise_stddev))
            np.random.seed(0)
            noise = np.random.normal(loc=0.0, scale=self.additive_noise_stddev, size=self.track_data.shape) * (self.track_data != self.nan_value)
            self.track_data += noise

        # csv data is aligned?
        self.data_is_aligned = data_is_aligned
        if data_is_aligned:
            self.aligned_track_data = self.track_data[:, :self.longest_track, :]
        else:
            logging.info("align data")
            self.aligned_track_data = self._convert_tracks_to_aligned_tracks(self.track_data)
            logging.info("data is aligned")
        # Create MLP data for tracking
        self.mlp_data, self.mlp_id_track_num, self.track_num_mlp_id = self._convert_aligned_tracks_to_mlp_data(self.aligned_track_data, n_inp_points = mlp_input_dim)
        # Create seq2seq data for tracking
        self.seq2seq_data = self._convert_aligned_tracks_to_seq2seq_data(self.aligned_track_data)
        # if data is aligned -> we create an artificial ordering of the tracks according to the given
        #  number of timesteps
        if data_is_aligned:
            self.birth_rate_mean = birth_rate_mean
            self.birth_rate_std = birth_rate_std
            self.artificial_tracks = self._expand_time_of_tracks(self.aligned_track_data)

        # normalize in all dimensions with the same factor
        if normalization_constant is None:
            # this leaves room for tracks at the borders
            self.normalization_constant = np.nanmax(self.seq2seq_data) * 1.1
        else:
            self.normalization_constant = normalization_constant
        self.belt_width = self.normalization_constant
        self.belt_height = self.belt_width
        # Create data for separation prediction
        if is_separation_prediction:
            self.separation_track_data, self.separation_spatial_labels, self.separation_temporal_labels = self.get_separation_prediction_data(
                virtual_belt_edge_x_position=virtual_belt_edge_x_position,
                virtual_nozzle_array_x_position=virtual_nozzle_array_x_position,
                min_measurements_count = min_measurements_count)

    def _load_tracks(self, data_is_aligned=True, rotate_columns=False, min_measurements_count=3):
        """Load tracks from csv files.

        Sets variables:
            self.timesteps:     sum of track_length over all files
            self.longest_track: Longest overall track in all files

        Args:
            data_is_aligned (Boolean):  Every track starts at t=0. There is no information about track start times provided.
            rotate_columns (Boolean):   File structure if False: [x, y] | if True: [y, x]
        
        Returns:
            track_data (np.array): All tracks. Shape: [n_tracks, timesteps, 2]
        """
        tracks = np.array([])
        n_prev = 0
        # Itearate over all files
        for file_ in self.file_list:
            # read the tracks from one session
            df = pd.read_csv(file_, engine='python')
            # We want to use 0.0 as NaN value. Therefore we have to check that it does not
            #   exist in the data.   Note: the double .min().min() is necessary because we
            #   first get the column minima and then we get the table minimum from that
            assert df.min().min() > 0.0, "Error: The dataframe {} contains a minimum <= 0.0".format(file_)
             # remove columns with less then n detections to filter out hard outliers
            df = df.dropna(axis=1, thresh=min_measurements_count, inplace=False)
            # Convert 2D df array to 3D numpy
            n_row = df.shape[0]
            n_col = df.shape[1]
            aranged_ids = np.arange(0, n_col)
            if rotate_columns:
                x_ids = aranged_ids%2==0
                y_ids = aranged_ids%2==1
            else:
                x_ids = aranged_ids%2==1
                y_ids = aranged_ids%2==0
            
            np_tracks = np.zeros([int(n_col/2), n_prev + n_row, 2])
            if data_is_aligned:
                # Add the tracks to the beginning
                np_tracks[:,:n_row,0] = df.iloc[:, x_ids].to_numpy(copy=True).swapaxes(0,1)
                np_tracks[:,:n_row,1] = df.iloc[:, y_ids].to_numpy(copy=True).swapaxes(0,1)
            else:
                # Add the tracks to the end
                np_tracks[:,n_prev:,0] = df.iloc[:, x_ids].to_numpy(copy=True).swapaxes(0,1)
                np_tracks[:,n_prev:,1] = df.iloc[:, y_ids].to_numpy(copy=True).swapaxes(0,1)
            # append new tracks to total tracks
            if tracks.shape[0]==0:
                # First file
                tracks = np.nan_to_num(np_tracks)
            else:
                # Add nan values (0.0) to end of existing tracks
                #tracks = np.concatenate((tracks, np.zeros([tracks.shape[0], n_row, 2]), axis=1)
                tracks = np.append(tracks, np.zeros([tracks.shape[0], n_row, 2]), axis=1)
                # Add the new tracks (and replace nan values with 0.0)
                tracks = np.append(tracks, np.nan_to_num(np_tracks), axis=0)

            n_prev += n_row

        self.timesteps = tracks.shape[1]
        # Get longest track and add 1 (?For some reasons?)
        self.longest_track = np.max(np.sum(tracks > 0, axis=1)) + 1
        return tracks

    def get_particle_timestep_data(self):
        if self.data_is_aligned:
            source = self.artificial_tracks
        else:
            source = self.get_track_data()
        particle_time_list = []
        # Iterate over all (artificial) timesteps in the data
        for t in range(source.shape[1]):
            all_particles = source[:, t]
            existing_particles = np.all(all_particles[:] != 0.0, axis=1)
            particle_time_list.append(np.append(np.where(existing_particles)[0][:,None], all_particles[existing_particles]/self.normalization_constant, axis=1))
        return particle_time_list


    def get_measurement_at_timestep(self, timestep, normalized=True):
        if self.data_is_aligned:
            source = self.artificial_tracks
        else:
            source = self.get_track_data()

        data = source[:, [timestep], :].copy()
        if normalized:
            return self.normalize_tracks(data, is_seq2seq_data=False)
        return data

    def _expand_time_of_tracks(self, aligned_tracks):
        """Given aligned_track => tracks for data association

        until all tracks are used:
            n = Normal.sample(birth_rate_mean, birth_rate_std)
            n = round(n)
            insert n new tracks at timestep
        """
        track_i = 0
        num_tracks = aligned_tracks.shape[0]
        time_step = 0

        tracks_beginning = []
        tracks_data = []

        while track_i < num_tracks:
            # how many new tracks (similar to half normal distribution)
            n_new_tracks = np.random.normal(0, 0.5 * self.birth_rate_std) + self.birth_rate_mean
            n_new_tracks = max(round(n_new_tracks), 0)

            # add all new tracks. Or: only as many as possible at the end
            for i in range(min(n_new_tracks, num_tracks - track_i)):
                tracks_beginning.append(time_step)
                tracks_data.append(aligned_tracks[track_i])
                track_i += 1

            time_step += 1

        # add enough white space at the end
        time_step += self.longest_track

        expanded_tracks = []

        for track_begin, track in zip(tracks_beginning, tracks_data):
            background = np.zeros([time_step, self.input_dim])
            background[track_begin:track_begin + track.shape[0], :] = track

            expanded_tracks.append(background.copy())

        expanded_tracks = np.array(expanded_tracks)
        self.timesteps = expanded_tracks.shape[1]
        return expanded_tracks


if __name__ == '__main__':
    f = FakeDataSet()
    f.plot_random_tracks()
