import code

import logging
import math
import os

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scipy.special import erf, erfinv
from scipy.stats import chi2
from scipy.stats.stats import pearsonr

from tensorflow.keras import backend as K
# issue with eager execution
# https://github.com/tensorflow/tensorflow/issues/34201
from tensorflow.python.keras import backend as K2

from sklearn.isotonic import IsotonicRegression
from matplotlib.collections import LineCollection

tf.keras.backend.set_floatx('float64')

rnn_models = {
    'lstm': tf.keras.layers.LSTM,
    'rnn': tf.keras.layers.SimpleRNN,
    'gru': tf.keras.layers.GRU
}


def rnn_model_factory(
        num_units_first_rnn=1024, num_units_second_rnn=16, num_units_third_rnn=0, num_units_fourth_rnn=0,
        num_units_first_dense=0, num_units_second_dense=0, num_units_third_dense=0, num_units_fourth_dense=0,
        rnn_model_name='lstm', dropout=0.0, regularization=0.0,
        use_batchnorm_on_dense=False, dropout_on_dense=False,
        num_time_steps=35, batch_size=128, nan_value=0, input_dim=2, output_dim=2,
        unroll=True, stateful=True):
    """
    Create a new keras model with the sequential API

    Note: The number of layers is fixed, because the HParams Board can't handle lists at the moment (Nov-2019)

    :param num_units_first_rnn:
    :param num_units_second_rnn:
    :param num_units_third_rnn:
    :param num_units_fourth_rnn:
    :param num_units_first_dense:
    :param num_units_second_dense:
    :param num_units_third_dense:
    :param num_units_fourth_dense:
    :param rnn_model_name:
    :param use_batchnorm_on_dense:
    :param num_time_steps:
    :param batch_size:
    :param nan_value:
    :param input_dim:
    :param output_dim:
    :param unroll:
    :param stateful: If this is False, then the state of the rnn will be reset after every batch.
            We want to control this manually therefore the default is True.
    :return: the model and a hash string identifying the architecture uniquely
    """
    model = tf.keras.Sequential()

    # hash for the model architecture
    hash_ = 'v1'

    # The Masking makes the model ignore time steps where the whole vector
    # consists of the *mask_value*
    model.add(tf.keras.layers.Masking(mask_value=nan_value, name="masking_layer",
                                      batch_input_shape=(batch_size, num_time_steps, input_dim)))
    hash_ += "-masking"

    # get the rnn layer
    rnn_model = rnn_models[rnn_model_name]

    # Add the recurrent layers
    rnn_layer_count = 0
    for hidden_units_in_rnn in [num_units_first_rnn, num_units_second_rnn, num_units_third_rnn, num_units_fourth_rnn]:
        if hidden_units_in_rnn == 0:
            # as soon as one layer has no units, we don't create the layer.
            break
        else:
            hash_ += "-{}[{}]".format(rnn_model_name, hidden_units_in_rnn)
            model.add(rnn_model(hidden_units_in_rnn,
                                return_sequences=True,
                                recurrent_dropout=dropout,
                                dropout=dropout,
                                stateful=stateful,
                                name='rnn-{}'.format(rnn_layer_count),
                                recurrent_initializer='glorot_uniform',
                                unroll=unroll,
                                recurrent_regularizer=tf.keras.regularizers.l2(regularization),
                                bias_regularizer=tf.keras.regularizers.l2(regularization),
                                kernel_regularizer=tf.keras.regularizers.l2(regularization)
                                )
                      )
            rnn_layer_count += 1

    # Add the dense layers
    for units_in_dense_layer in [num_units_first_dense, num_units_second_dense, num_units_third_dense,
                                 num_units_fourth_dense]:
        if units_in_dense_layer == 0:
            # as soon as one layer has no units, we don't create the layer, neither further ones
            break
        else:
            hash_ += "-dense[{}, leakyrelu]".format(units_in_dense_layer)
            if dropout > 0.0 and dropout_on_dense:
                model.add(tf.keras.layers.Dropout(dropout))
            model.add(tf.keras.layers.Dense(units_in_dense_layer,
                                            bias_regularizer=tf.keras.regularizers.l2(regularization),
                                            kernel_regularizer=tf.keras.regularizers.l2(regularization)
                                            ))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            if use_batchnorm_on_dense:
                model.add(tf.keras.layers.BatchNormalization())
                hash_ += "-BatchNorm"

    # Always end with a dense layer with
    # - two outputs (x, y)
    # - or: three outputs (x, y, y_separation)
    # - or: four outputs (x, y, y_separation, t_separation)
    model.add(tf.keras.layers.Dense(output_dim,
                                    bias_regularizer=tf.keras.regularizers.l2(regularization),
                                    kernel_regularizer=tf.keras.regularizers.l2(regularization)
                                    ))

    hash_ += "-dense[{}]".format(output_dim)

    return model, hash_


def train_step_separation_prediction_generator(model,
                                               optimizer, batch_size, num_time_steps, nan_value=0,
                                               time_normalization=22., only_last_timestep_additional_loss=True,
                                               apply_gradients=True):
    # the placeholder character used for padding
    mask_value = K.variable(np.array([nan_value, nan_value]), dtype=tf.float64)

    input_dim = 2

    no_loss_mask_np = np.ones([batch_size, num_time_steps])
    # no_loss_mask_np[:, :7] = 0
    no_loss_mask = tf.constant(no_loss_mask_np)

    # time label should be relative not absolute: instead of predicting the
    #  frame number when the particle crosses the bar, we make a countdown.
    #  example: time_label = [4, 4, 4, 4, 4, 4]
    #  new_time_label = [4, 3, 2, 1, 0, -1]
    # -> calc like this: range = [0, 1, 2, 3, 4, 5 ...]
    #     new_time_label = time_label - range
    range_np = np.array(list(range(num_time_steps))) / time_normalization
    range_np_batch = np.tile(range_np, (batch_size, 1)).reshape([batch_size, num_time_steps, 1])
    range_np_batch_2 = np.tile(range_np, (batch_size, 1)).reshape([batch_size, num_time_steps])
    range_ = K.variable(range_np_batch)
    range_2 = K.variable(range_np_batch_2)

    @tf.function
    def train_step(inp, target):
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            predictions = model(inp)

            mask = K.all(K.equal(inp, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            pred_mse = tf.keras.losses.mean_squared_error(target[:, :, :2], predictions[:, :, :2]) * mask
            pred_mse = K.sum(pred_mse) / K.sum(mask)

            pred_mae = tf.keras.losses.mean_absolute_error(target[:, :, :2], predictions[:, :, :2]) * mask
            pred_mae = K.sum(pred_mae) / K.sum(mask)

            # find the last timestep for every track
            # 1. find timesteps which equal [0,0]
            # 2. get the first timestep for every track
            # 2.1 [0,0,0,1,1,1]
            # 2.2 [1,1,1,0,0,0] * length**2 + [0,0,0,1,1,1] * range(length)
            # 2.3 argmin
            # 3 Create a mask from that

            if only_last_timestep_additional_loss:
                any_zero_element = K.equal(inp, [nan_value])
                any_zero_element = K.cast(any_zero_element, tf.float64)
                count_zeros_per_timestep = K.sum(any_zero_element, axis=-1)
                two_zeros_per_timestep = K.equal(count_zeros_per_timestep, [input_dim])
                is_zero = K.cast(two_zeros_per_timestep, tf.float64)
                not_zero = (-1 * is_zero + 1) * (inp.shape[1] ** 2)
                search_space = is_zero * range_2 + not_zero
                last_timesteps = K.argmin(search_space, axis=1) - 1

                mask_last_step = tf.one_hot(last_timesteps, depth=inp.shape[1], dtype=tf.bool, on_value=True,
                                            off_value=False)
                mask_last_step = K.cast(mask_last_step, tf.float64)

            start = None
            end = None

            spatial_mse = tf.keras.losses.mean_squared_error(target[:, start:end, 2:3],
                                                             predictions[:, start:end, 2:3]) * mask
            spatial_mae = tf.keras.losses.mean_absolute_error(target[:, start:end, 2:3],
                                                              predictions[:, start:end, 2:3]) * mask

            if only_last_timestep_additional_loss:
                spatial_mse = K.sum(spatial_mse * mask_last_step) / batch_size
                spatial_mae = K.sum(spatial_mae * mask_last_step) / batch_size
            else:
                spatial_mse = K.sum(spatial_mse) / K.sum(mask)
                spatial_mae = K.sum(spatial_mae) / K.sum(mask)

            # new_time_target is like a countdown
            new_time_target = (target[:, :, 3:4] - range_)

            temporal_mse = tf.keras.losses.mean_squared_error(new_time_target[:, start:end],
                                                              predictions[:, start:end, 3:4]) * mask
            temporal_mae = tf.keras.losses.mean_squared_error(new_time_target[:, start:end],
                                                              predictions[:, start:end, 3:4]) * mask

            if only_last_timestep_additional_loss:
                temporal_mse = K.sum(temporal_mse * mask_last_step) / batch_size
                temporal_mae = K.sum(temporal_mae * mask_last_step) / batch_size
            else:
                temporal_mse = K.sum(temporal_mse) / K.sum(mask)
                temporal_mae = K.sum(temporal_mae) / K.sum(mask)

            mse = pred_mse + spatial_mse + temporal_mse + tf.add_n(model.losses)
            mae = pred_mae + spatial_mae + temporal_mae + tf.add_n(model.losses)

        if apply_gradients:
            grads = tape.gradient(mse, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return mse, mae, pred_mse, pred_mae, spatial_mse, spatial_mae, temporal_mse, temporal_mae

    return train_step


def train_kendall_step_generator(model, optimizer, nan_value=0.0):
    mask_value = K.variable(np.array([nan_value, nan_value]), dtype=tf.float64)

    @tf.function
    def train_step(inp, target):
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            predictions = model(inp)

            mask = K.all(K.equal(target, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            target_x = target[:, :, 0]
            target_y = target[:, :, 1]

            pos_pred_x = predictions[:, :, 0]
            pos_pred_y = predictions[:, :, 1]

            log_var_pred_x = predictions[:, :, 2]
            log_var_pred_y = predictions[:, :, 3]

            pos_loss_x = (target_x - pos_pred_x)**2

            bnn_loss_x = 0.5 * K.exp(-log_var_pred_x) * pos_loss_x + 0.5 * log_var_pred_x

            pos_loss_y = (target_y - pos_pred_y)**2
            bnn_loss_y = 0.5 * K.exp(-log_var_pred_y) * pos_loss_y + 0.5 * log_var_pred_y

            neg_log_likelihood = mask * (bnn_loss_x + bnn_loss_y)

            neg_log_likelihood = K.sum(neg_log_likelihood) / K.sum(mask) + tf.add_n(model.losses)

            # following lines are for logging metric
            mse = tf.keras.losses.mean_squared_error(target, predictions[:, :, :2]) * mask
            mae = tf.keras.losses.mean_absolute_error(target, predictions[:, :, :2]) * mask
            mse = K.sum(mse) / K.sum(mask) + tf.add_n(model.losses)
            mae = K.sum(mae) / K.sum(mask) + tf.add_n(model.losses)

        grads = tape.gradient(neg_log_likelihood, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return mse, mae, neg_log_likelihood

    return train_step


def train_custom_variance_step_generator(model, optimizer, nan_value=0.0):
    mask_value = K.variable(np.array([nan_value, nan_value]), dtype=tf.float64)

    @tf.function
    def train_step(inp, target):
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            predictions = model(inp)

            mask = K.all(K.equal(target, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            target_x = target[:, :, 0]
            target_y = target[:, :, 1]

            pos_pred_x = predictions[:, :, 0]
            pos_pred_y = predictions[:, :, 1]

            log_var_pred_x = predictions[:, :, 2]
            log_var_pred_y = predictions[:, :, 3]

            pos_loss_x = (target_x - pos_pred_x)**2
            pos_loss_y = (target_y - pos_pred_y)**2

            var_loss_x = (pos_loss_x - K.exp(log_var_pred_x))**2
            var_loss_y = (pos_loss_y - K.exp(log_var_pred_y))**2

            custom_loss = mask * (pos_loss_x + pos_loss_y + var_loss_x + var_loss_y)

            custom_loss = K.sum(custom_loss) / K.sum(mask) + tf.add_n(model.losses)

            # following lines are for logging metric
            mse = tf.keras.losses.mean_squared_error(target, predictions[:, :, :2]) * mask
            mae = tf.keras.losses.mean_absolute_error(target, predictions[:, :, :2]) * mask
            mse = K.sum(mse) / K.sum(mask) + tf.add_n(model.losses)
            mae = K.sum(mae) / K.sum(mask) + tf.add_n(model.losses)

        grads = tape.gradient(custom_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return mse, mae, custom_loss

    return train_step


def train_step_generator(model, optimizer, nan_value=0.0):
    """
    Build a function which returns a computational graph for tensorflow.
    This function can be called to train the given model with the given
    optimizer.

    :param model: model according to estimator api
    :param optimizer: tf estimator
    :param nan_value: e.g. 0

    Example:
        >>> import data_manager
        >>> keras_model = rnn_model_factory()
        >>> dataset_train, _ = data_manager.FakeDataSet().get_tf_data_sets_seq2seq_data()
        >>> train_step = model.train_step_generator(model, optimizer)
        >>>
        >>> step, batch_size = 0, 32
        >>> for epoch in range(100):
        >>>     for (batch_n, (inp, target)) in enumerate(dataset_train):
        >>>         # _ = keras_model.reset_states()
        >>>         mse, mae = train_step(inp, target)
        >>>         step += batch_size
        >>>     tf.summary.scalar('loss', mse, step=step)

    :return: function which can be called to train the given model with the given optimizer
    """
    # the placeholder character used for padding
    mask_value = K.variable(np.array([nan_value, nan_value]), dtype=tf.float64)

    @tf.function
    def train_step(inp, target):
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            predictions = model(inp)

            mask = K.all(K.equal(target, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            # mse = tf.keras.losses.mean_squared_error(target, predictions) * mask
            # mae = tf.keras.losses.mean_absolute_error(target, predictions) * mask
            se_x = ((target[:, :, 0] - predictions[:, :, 0]) ** 2)
            se_y = ((target[:, :, 1] - predictions[:, :, 1]) ** 2)
            se = (se_x+se_y) * mask

            ae_x = ((target[:, :, 0] - predictions[:, :, 0]) ** 2) ** 0.5
            ae_y = ((target[:, :, 1] - predictions[:, :, 1]) ** 2) ** 0.5
            ae = (ae_x + ae_y) * mask

            # take average w.r.t. the number of unmasked entries
            mse = K.sum(se) / K.sum(mask) + tf.add_n(model.losses)
            mae = K.sum(ae) / K.sum(mask) + tf.add_n(model.losses)

        grads = tape.gradient(mse, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return mse, mae

    return train_step


def train_epoch_generator(rnn_model, train_step, dataset_train, batch_size):
    """
    Build a function which returns a computational graph for tensorflow.
    This function can be called to train the given model with the given
    optimizer **for a number of epochs**.
    It optimizes the training as it lets tensorflow train whole epochs
    without coming back to the python code.

    This uses the train_step. You can call the train_step function on
    your own if you want more control.

    Attention: Keyboard interrupts are not stopping the training.

    Example:
        >>> import data
        >>> dataset_train, _ = data.FakeDataSet().get_tf_data_sets_seq2seq_data()
        >>> train_step = rnn_model.train_step_generator(rnn_model,  tf.keras.optimizers.Adam())
        >>> step, batch_size = 0, 32
        >>> for epoch in range(100):
        >>>     avg_loss, train_step_counter = train_epoch(1)
        >>>     tf.summary.scalar('loss', avg_loss, step=train_step_counter)

    :param rnn_model:
    :param train_step:
    :param dataset_train:
    :param batch_size:
    :return:
    """

    @tf.function
    def train_epoch(n_epochs):
        train_step_counter = tf.constant(0, dtype=tf.float64)
        batch_counter = tf.constant(0, dtype=tf.float64)
        sum_loss = tf.constant(0, dtype=tf.float64)

        for inp, target in dataset_train.repeat(n_epochs):
            hidden = rnn_model.reset_states()
            loss = train_step(inp, target)
            sum_loss += loss
            train_step_counter += batch_size
            batch_counter += 1

        avg_loss = sum_loss / batch_counter

        return avg_loss, train_step_counter

    return train_epoch


def set_state(rnn_model, batch_state):
    rnn_layer_counter = 0
    for i in range(1000):
        try:
            layer = rnn_model.get_layer(index=i)
        except:
            break

        if isinstance(layer, tf.keras.layers.RNN):
            for sub_state_number, sub_state in enumerate(layer.states):
                layer.states[sub_state_number].assign(
                    tf.convert_to_tensor(batch_state[rnn_layer_counter][sub_state_number]))
            rnn_layer_counter += 1


def get_state(rnn_model):
    rnn_layer_states = []
    # get all layers in ascending order
    i = -1
    while True:
        i += 1
        # Not asking for permission but handling the error is faster in python
        try:
            layer = rnn_model.get_layer(index=i)
        except:
            break

        # only store the state of the layer if it is a recurrent layer
        #   DenseLayers don't have a state
        if isinstance(layer, tf.keras.layers.RNN):
            rnn_layer_states.append([sub_state.numpy() for sub_state in layer.states])

    return rnn_layer_states


class Model(object):
    def __init__(self, global_config, data_source):
        self.global_config = global_config
        self.data_source = data_source

        self.is_calibrated = False
        self._iso_regression_fn = None

        # ToDo: implement uncertainties + separation prediction
        self._label_dim = 2
        if self.global_config['kendall_loss']:
            self._label_dim += 2
        if self.global_config['separation_prediction']:
            self._label_dim += 2
        if self.global_config['custom_variance_prediction']:
            self._label_dim += 2

        if self.global_config['is_loaded']:
            self.rnn_model = tf.keras.models.load_model(self.global_config['model_path'])
            logging.info(self.rnn_model.summary())
            self.rnn_model.reset_states()
        else:
            if self.global_config['kendall_loss']:
                self.train_kendall()
            elif self.global_config['custom_variance_prediction']:
                self.train_custom_variance()
            elif self.global_config['separation_prediction']:
                self.train_separation_prediction()
            else:
                self.train()

        if self.global_config['calibrate']:
            logging.info("Calibrate")
            self.calibrate()

    def calibrate(self):
        if self.global_config['calibrate']:
            self._calibrate()
            self.is_calibrated = True

    @staticmethod
    def conf_to_sigma(confs, sigma=1.0):
        return chi2.ppf(confs, df=2) * sigma
        # return erfinv(confs) * np.sqrt(2)

    def _apply_isotonic_regression(self, confs):
        if self._iso_regression_fn is not None:
            return self._iso_regression_fn.predict(confs)
        else:
            logging.info('iso regression function was none')
            return confs

    def _calibrate(self):
        dataset_train, dataset_test = self.data_source.get_tf_data_sets_seq2seq_data(normalized=True, seed=42)
        data = self._get_evaluation_data(dataset_test)
        N = data['time_step'].shape[0]

        expected_confs = []
        cdf = []
        counter = []

        for expected_confidence in np.arange(0.0, 0.9999, 0.01):
            expected_1_sigma_distance = chi2.ppf(expected_confidence, df=2)

            count_falls_into = np.count_nonzero(data['standardized_l2'] <= expected_1_sigma_distance)
            counter.append(count_falls_into)
            proportion = count_falls_into / N
            cdf.append(proportion)
            expected_confs.append(expected_confidence)

        expected_confs = np.array(expected_confs)
        cdf = np.array(cdf)

        x = expected_confs
        y = cdf
        n = x.shape[0]

        ir = IsotonicRegression()
        y_ = ir.fit_transform(x, y)
        y_pred = ir.predict(x)

        fig, ax = plt.subplots(ncols=1)
        ax1 = ax.twinx()
        ax.hist(expected_confs, weights=counter, density=False, bins=50, histtype='stepfilled', alpha=0.2)
        ax.set_ylabel("# Predictions in conf. interval")

        ax1.scatter(expected_confs, cdf, c='blue')
        ax1.plot(expected_confs, y_pred, c='black')
        plt.title("Calibration Plot")

        ax1.set_xlabel("Expected confidence level")
        ax1.set_ylabel("Empirical confidence level")



        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'Calibration.pdf'))
        plt.clf()

        # x and y swapped

        self._iso_regression_fn = IsotonicRegression(out_of_bounds='clip')
        self._iso_regression_fn.fit_transform(y, x)
        x_pred = self._iso_regression_fn.predict(y)

        plt.scatter(cdf, expected_confs, c='blue')
        plt.plot(cdf, x_pred, c='black')

        plt.title("Flipped Calibration plot (standardized euclidean distance)")
        plt.xlabel("Observed confidence level")
        plt.ylabel("Expected confidence level")
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'CalibrationFlipped.pdf'))
        plt.clf()

        # Plot correlations
        self.plot_correlation(None, data=data)

    def get_zero_state(self):
        self.rnn_model.reset_states()
        return get_state(self.rnn_model)

    # expected to return list<vector<pair<float,float>>>, list<RNNStateTuple>
    def predict(self, current_input, state, track_measurement_history=None):
        new_states = []
        predictions = []

        if self.global_config['mc_dropout'] and self.global_config['mc_samples'] > 1:
            samples = []

            k = self.global_config['mc_samples']

            # Unfortunately we cannot use the state because dropout at inference time changes and therefore the
            #  state would be a bottleneck. For this reason, we use the measurement history of the tracks.
            data_input = np.ones([self.global_config['batch_size'], self.data_source.longest_track, 2]) * \
                         self.data_source.nan_value

            last_timestep_per_track = []

            for track_i, track_history in enumerate(track_measurement_history):
                # use maximum of timesteps
                t = 0
                offset = max(0, len(track_history) - self.data_source.longest_track)
                for i in range(min(len(track_history), self.data_source.longest_track)):
                    data_input[track_i, t, 0] = track_history[i+offset][0]
                    data_input[track_i, t, 1] = track_history[i+offset][1]
                    t += 1
                last_timestep_per_track.append(t-1)

            for _ in range(k):
                self.rnn_model.reset_states()
                # set_state(self.rnn_model, state)

                # https://github.com/tensorflow/tensorflow/issues/34201
                # ... eager execution bug with keras functions
                # mitigation: set learning_phase=1  =>  dropout is active
                # predic = f((x_input,))[0]

                # K2.set_learning_phase(1)
                predic = self.rnn_model(data_input, training=True)

                # extract last prediction (they are the only predictions which we need)
                predictions = np.zeros([self.global_config['batch_size'], 1, 2])
                for track_i, timestep in enumerate(last_timestep_per_track):
                    predictions[track_i, 0] = predic[track_i, timestep]

                samples.append(predictions)

            samples = np.array(samples)
            sample_mean = np.sum(samples, axis=0) / float(k)
            sample_variance = np.sum((samples - sample_mean) ** 2, axis=0) / float(k)

            prediction = sample_mean
            variances = sample_variance

        elif self.global_config['mc_dropout'] and self.global_config['mc_samples'] == 1:
            current_input = np.expand_dims(current_input, axis=1)

            _ = self.rnn_model.reset_states()
            prediction = self.rnn_model(current_input, training=False).numpy()
            variances = np.ones_like(prediction) * 0.01

        elif self.global_config['kendall_loss']:
            current_input = np.expand_dims(current_input, axis=1)
            set_state(self.rnn_model, state)
            prediction_and_variance = self.rnn_model(current_input).numpy()

            prediction = np.copy(prediction_and_variance[:, :, :2])
            # the network outputs: log(variance) -> therefore we apply the exponential function to the result
            variances = np.copy(K.exp(prediction_and_variance[:, :, 2:4]))

        else:
            current_input = np.expand_dims(current_input, axis=1)
            set_state(self.rnn_model, state)
            prediction = self.rnn_model(current_input)
            variances = None

        # ToDo: use the separation predictions! Currently I just drop them.
        if self.global_config['separation_prediction']:
            prediction = np.copy(prediction.numpy()[:, :, :2])

        if variances is not None:
            variances = np.squeeze(variances)

        prediction = np.squeeze(prediction)
        new_state = get_state(self.rnn_model)
        return prediction, new_state, variances

    def train_kendall(self):
        """Train  Heteroscedastic Aleatoric Uncertainty (https://arxiv.org/pdf/1703.04977.pdf)"""
        self.rnn_model, self.model_hash = rnn_model_factory(batch_size=self.global_config['batch_size'],
                                                            num_time_steps=self.data_source.longest_track,
                                                            output_dim=self._label_dim,
                                                            **self.global_config['rnn_model_factory'])
        self.global_config['rnn_time_steps'] = self.data_source.longest_track
        logging.info(self.rnn_model.summary())

        dataset_train, dataset_test = self.data_source.get_tf_data_sets_seq2seq_data(normalized=True)

        optimizer = tf.keras.optimizers.Adam()

        train_step_fn = train_kendall_step_generator(self.rnn_model, optimizer)

        train_losses = []
        test_losses = []

        # Train model
        for epoch in range(self.global_config['num_train_epochs']):
            # learning rate decay
            if (epoch + 1) % self.global_config['lr_decay_after_epochs'] == 0:
                old_lr = K.get_value(optimizer.lr)
                new_lr = old_lr * self.global_config['lr_decay_factor']
                logging.info("Reducing learning rate from {} to {}.".format(old_lr, new_lr))
                K.set_value(optimizer.lr, new_lr)

            # Train for one batch
            mae_batch = []
            mse_batch = []
            neg_log_likelihood_batch = []

            _ = self.rnn_model.reset_states()
            for (batch_n, (inp, target)) in enumerate(dataset_train):
                # Mini-Batches
                if self.global_config['clear_state']:
                    _ = self.rnn_model.reset_states()
                mse, mae, neg_log_likelihood = train_step_fn(inp, target)
                mse_batch.append(mse)
                mae_batch.append(mae)
                neg_log_likelihood_batch.append(neg_log_likelihood)

            mse = np.mean(mse_batch)
            mae = np.mean(mae_batch)
            neg_log_likelihood = np.mean(neg_log_likelihood_batch)

            train_losses.append([epoch, mse, mae * self.data_source.normalization_constant, neg_log_likelihood])

            log_string = "{}/{}: \t nll={} \t mse={}".format(epoch, self.global_config['num_train_epochs'],
                                                             neg_log_likelihood, mse)

            # Evaluate
            if (epoch + 1) % self.global_config['evaluate_every_n_epochs'] == 0 \
                    or (epoch + 1) == self.global_config['num_train_epochs']:
                logging.info(log_string)
                test_mse, test_mae, test_nll = self._evaluate_kendall_model(dataset_test, epoch)
                test_losses.append([epoch, test_mse, test_mae * self.data_source.normalization_constant, test_nll])
                self.plot_track_with_uncertainty(dataset_test, epoch=epoch)
                self.plot_calibration(dataset_test, epoch=epoch)
                self.plot_correlation(dataset_test, epoch=epoch)
            else:
                logging.debug(log_string)

        self.rnn_model.save(os.path.join(self.global_config['diagrams_path'], 'model.h5'))

        # Visualize loss curve
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # MSE
        plt.plot(train_losses[:, 0], train_losses[:, 1], c='blue', label="Training MSE")
        plt.plot(test_losses[:, 0], test_losses[:, 1], c='red', label="Test MSE")
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MSE.pdf'))
        plt.clf()

        # MAE
        plt.plot(train_losses[:, 0], train_losses[:, 2], c='blue', label="Training MAE (not normalized)")
        plt.plot(test_losses[:, 0], test_losses[:, 2], c='red', label="Test MAE (not normalized)")
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MAE.pdf'))
        plt.clf()

        # NLL
        plt.plot(train_losses[:, 0], train_losses[:, 3], c='blue', label="Training Negative Log Likelihood")
        plt.plot(test_losses[:, 0], test_losses[:, 3], c='red', label="Test Negative Log Likelihood")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'NLL.pdf'))
        plt.clf()

    def train_custom_variance(self):
        self.rnn_model, self.model_hash = rnn_model_factory(batch_size=self.global_config['batch_size'],
                                                            num_time_steps=self.data_source.longest_track,
                                                            output_dim=self._label_dim,
                                                            **self.global_config['rnn_model_factory'])
        self.global_config['rnn_time_steps'] = self.data_source.longest_track
        logging.info(self.rnn_model.summary())

        dataset_train, dataset_test = self.data_source.get_tf_data_sets_seq2seq_data(normalized=True)

        optimizer = tf.keras.optimizers.Adam()

        train_step_fn = train_custom_variance_step_generator(self.rnn_model, optimizer)

        train_losses = []
        test_losses = []

        # Train model
        for epoch in range(self.global_config['num_train_epochs']):
            # learning rate decay
            if (epoch + 1) % self.global_config['lr_decay_after_epochs'] == 0:
                old_lr = K.get_value(optimizer.lr)
                new_lr = old_lr * self.global_config['lr_decay_factor']
                logging.info("Reducing learning rate from {} to {}.".format(old_lr, new_lr))
                K.set_value(optimizer.lr, new_lr)

            # Train for one batch
            mae_batch = []
            mse_batch = []
            custom_loss_batch = []

            _ = self.rnn_model.reset_states()
            for (batch_n, (inp, target)) in enumerate(dataset_train):
                # Mini-Batches
                if self.global_config['clear_state']:
                    _ = self.rnn_model.reset_states()
                mse, mae, custom_loss = train_step_fn(inp, target)
                mse_batch.append(mse)
                mae_batch.append(mae)
                custom_loss_batch.append(custom_loss)

            mse = np.mean(mse_batch)
            mae = np.mean(mae_batch)
            custom_loss = np.mean(custom_loss_batch)

            train_losses.append([epoch, mse, mae * self.data_source.normalization_constant, custom_loss])

            log_string = "{}/{}: \t custom_loss={} \t mse={}".format(epoch, self.global_config['num_train_epochs'],
                                                             custom_loss, mse)

            # Evaluate
            if (epoch + 1) % self.global_config['evaluate_every_n_epochs'] == 0 \
                    or (epoch + 1) == self.global_config['num_train_epochs']:
                logging.info(log_string)
                test_mse, test_mae, test_custom_loss = self._evaluate_custom_loss_model(dataset_test, epoch)
                test_losses.append([epoch, test_mse, test_mae * self.data_source.normalization_constant, test_custom_loss])
                self.plot_track_with_uncertainty(dataset_test, epoch=epoch)
                self.plot_calibration(dataset_test, epoch=epoch)
                self.plot_correlation(dataset_test, epoch=epoch)
            else:
                logging.debug(log_string)

        self.rnn_model.save(os.path.join(self.global_config['diagrams_path'], 'model.h5'))

        # Visualize loss curve
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # MSE
        plt.plot(train_losses[:, 0], train_losses[:, 1], c='blue', label="Training MSE")
        plt.plot(test_losses[:, 0], test_losses[:, 1], c='red', label="Test MSE")
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MSE.pdf'))
        plt.clf()

        # MAE
        plt.plot(train_losses[:, 0], train_losses[:, 2], c='blue', label="Training MAE (not normalized)")
        plt.plot(test_losses[:, 0], test_losses[:, 2], c='red', label="Test MAE (not normalized)")
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MAE.pdf'))
        plt.clf()

        # custom_loss
        plt.plot(train_losses[:, 0], train_losses[:, 3], c='blue', label="Training custom_loss")
        plt.plot(test_losses[:, 0], test_losses[:, 3], c='red', label="Test custom_loss")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'Custom_loss.pdf'))
        plt.clf()

    def train(self):
        self.rnn_model, self.model_hash = rnn_model_factory(batch_size=self.global_config['batch_size'],
                                                            num_time_steps=self.data_source.longest_track,
                                                            output_dim=self._label_dim,
                                                            **self.global_config['rnn_model_factory'])
        # try to compile model
        # self.rnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse', metrics=['accuracy'])

        self.global_config['rnn_time_steps'] = self.data_source.longest_track
        logging.info(self.rnn_model.summary())

        dataset_train, dataset_test = self.data_source.get_tf_data_sets_seq2seq_data(normalized=True)

        optimizer = tf.keras.optimizers.Adam()
        train_step_fn = train_step_generator(self.rnn_model, optimizer)

        # dict(epoch->float)
        train_losses = []
        test_losses = []

        # Train model
        for epoch in range(self.global_config['num_train_epochs']):
            # learning rate decay after 100 epochs
            if (epoch + 1) % self.global_config['lr_decay_after_epochs'] == 0:
                old_lr = K.get_value(optimizer.lr)
                new_lr = old_lr * self.global_config['lr_decay_factor']
                logging.info("Reducing learning rate from {} to {}.".format(old_lr, new_lr))
                K.set_value(optimizer.lr, new_lr)

            # Train for one batch
            mae_batch = []
            mse_batch = []

            _ = self.rnn_model.reset_states()
            for (batch_n, (inp, target)) in enumerate(dataset_train):
                # Mini-Batches
                if self.global_config['clear_state']:
                    _ = self.rnn_model.reset_states()
                mse, mae = train_step_fn(inp, target)
                mse_batch.append(mse)
                mae_batch.append(mae)
            mse = np.mean(mse_batch)
            mae = np.mean(mae_batch)
            train_losses.append([epoch, mse, mae * self.data_source.normalization_constant])

            log_string = "{}/{}: \t loss={}".format(epoch, self.global_config['num_train_epochs'], mse)

            # Evaluate
            if (epoch + 1) % self.global_config['evaluate_every_n_epochs'] == 0 \
                    or (epoch + 1) == self.global_config['num_train_epochs']:
                logging.info(log_string)
                test_mse, test_mae = self._evaluate_model(dataset_test, epoch)
                test_losses.append([epoch, test_mse, test_mae * self.data_source.normalization_constant])
                if self.global_config['mc_dropout']:
                    self.plot_track_with_uncertainty(dataset_test, epoch=epoch)
                    self.plot_calibration(dataset_test, epoch=epoch)
                    self.plot_correlation(dataset_test, epoch=epoch)
                plt.close('all')
            else:
                logging.debug(log_string)

        self.rnn_model.save(os.path.join(self.global_config['diagrams_path'], 'model.h5'))

        # Visualize loss curve
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # MSE
        plt.plot(train_losses[:, 0], train_losses[:, 1], c='blue', label="Training MSE")
        plt.plot(test_losses[:, 0], test_losses[:, 1], c='red', label="Test MSE")
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MSE.pdf'))
        plt.clf()

        # MAE
        plt.plot(train_losses[:, 0], train_losses[:, 2], c='blue', label="Training MAE (not normalized)")
        plt.plot(test_losses[:, 0], test_losses[:, 2], c='red', label="Test MAE (not normalized)")
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MAE.pdf'))
        plt.clf()

        plt.close('all')

    def train_separation_prediction(self):
        dataset_train, dataset_test, num_time_steps = self.data_source.get_tf_data_sets_seq2seq_with_separation_data(
            normalized=True,
            time_normalization=self.global_config['time_normalization_constant'],
            virtual_belt_edge_x_position=self.global_config['virtual_belt_edge_x_position'],
            virtual_nozzle_array_x_position=self.global_config['virtual_nozzle_array_x_position']
        )

        self.rnn_model, self.model_hash = rnn_model_factory(batch_size=self.global_config['batch_size'],
                                                            num_time_steps=num_time_steps,
                                                            output_dim=self._label_dim,
                                                            **self.global_config['rnn_model_factory'])
        self.global_config['rnn_time_steps'] = num_time_steps
        logging.info(self.rnn_model.summary())

        optimizer = tf.keras.optimizers.Adam()
        train_step_fn = train_step_separation_prediction_generator(self.rnn_model, optimizer,
                                                                   batch_size=self.global_config['batch_size'],
                                                                   num_time_steps=num_time_steps,
                                                                   time_normalization=self.global_config[
                                                                       'time_normalization_constant'],
                                                                   only_last_timestep_additional_loss=
                                                                   self.global_config[
                                                                       'only_last_timestep_additional_loss'],
                                                                   apply_gradients=True
                                                                   )

        test_step_fn = train_step_separation_prediction_generator(self.rnn_model, optimizer,
                                                                  batch_size=self.global_config['batch_size'],
                                                                  num_time_steps=num_time_steps,
                                                                  time_normalization=self.global_config[
                                                                      'time_normalization_constant'],
                                                                  only_last_timestep_additional_loss=
                                                                  self.global_config[
                                                                      'only_last_timestep_additional_loss'],
                                                                  apply_gradients=False
                                                                  )

        # dict(epoch->float)
        train_losses = []
        test_losses = []

        # Train model
        epoch = 0
        for epoch in range(self.global_config['num_train_epochs']):
            # learning rate decay after 100 epochs
            if (epoch + 1) % self.global_config['lr_decay_after_epochs'] == 0:
                old_lr = K.get_value(optimizer.lr)
                new_lr = old_lr * self.global_config['lr_decay_factor']
                logging.info("Reducing learning rate from {} to {}.".format(old_lr, new_lr))
                K.set_value(optimizer.lr, new_lr)

            # Train for one batch
            errors_in_one_batch = []

            _ = self.rnn_model.reset_states()
            for (batch_n, (inp, target)) in enumerate(dataset_train):
                # Mini-Batches
                if self.global_config['clear_state']:
                    _ = self.rnn_model.reset_states()
                errors = train_step_fn(inp, target)
                errors_in_one_batch.append(errors)

            error_list = np.mean(errors_in_one_batch, axis=0).tolist()
            train_losses.append([epoch] + error_list)

            log_string = "{}/{}: \t MSE={}".format(epoch, self.global_config['num_train_epochs'], error_list[0])

            # Evaluate
            if (epoch + 1) % self.global_config['evaluate_every_n_epochs'] == 0 \
                    or (epoch + 1) == self.global_config['num_train_epochs']:
                logging.info(log_string)

                errors_in_one_batch = []

                _ = self.rnn_model.reset_states()
                for (batch_n, (inp, target)) in enumerate(dataset_test):
                    # Mini-Batches
                    if self.global_config['clear_state']:
                        _ = self.rnn_model.reset_states()
                    errors = test_step_fn(inp, target)
                    errors_in_one_batch.append(errors)

                error_list = np.mean(errors_in_one_batch, axis=0).tolist()
                test_losses.append([epoch] + error_list)
            else:
                logging.debug(log_string)

        self._evaluate_separation_model(dataset_test, epoch)

        # Store meta info
        self.rnn_model.save(os.path.join(self.global_config['diagrams_path'], 'model.h5'))

        # Visualize loss curve
        # columns: epoch, mse, mae, pred_mse, pred_mae, spatial_mse, spatial_mae, temporal_mse, temporal_mae
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # MSEs
        plt.plot(train_losses[:, 0], train_losses[:, 1], c='navy', label="Training MSE")
        plt.plot(train_losses[:, 0], train_losses[:, 3], c='blue', label="Training MSE (prediction)")
        plt.plot(train_losses[:, 0], train_losses[:, 5], c='cornflowerblue', label="Training MSE (sep. spatial)")
        plt.plot(train_losses[:, 0], train_losses[:, 7], c='deepskyblue', label="Training MSE (sep. temporal)")

        plt.plot(test_losses[:, 0], test_losses[:, 1], c='maroon', label="Test MSE")
        plt.plot(test_losses[:, 0], test_losses[:, 3], c='red', label="Test MSE (prediction)")
        plt.plot(test_losses[:, 0], test_losses[:, 5], c='tomato', label="Test MSE (sep. spatial)")
        plt.plot(test_losses[:, 0], test_losses[:, 7], c='orange', label="Test MSE (sep. temporal)")
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MSE.pdf'))
        plt.clf()

        # MAEs
        plt.plot(train_losses[:, 0], train_losses[:, 2], c='blue', label="Training MAE")
        plt.plot(test_losses[:, 0], test_losses[:, 2], c='red', label="Test MAE")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MAE.pdf'))
        plt.clf()

    def _evaluate_kendall_model(self, dataset_test, epoch):
        mses = np.array([])
        maes = np.array([])
        nlls = np.array([])

        mask_value = K.variable(np.array([self.data_source.nan_value, self.data_source.nan_value]), dtype=tf.float64)
        normalization_factor = self.data_source.normalization_constant

        _ = self.rnn_model.reset_states()
        for input_batch, target_batch in dataset_test:
            if self.global_config['clear_state']:
                _ = self.rnn_model.reset_states()

            batch_predictions = self.rnn_model(input_batch)

            # Calculate the mask
            mask = K.all(K.equal(target_batch, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            target_batch_unnormalized = target_batch
            pred_batch_unnormalized = batch_predictions

            batch_loss = tf.keras.losses.mean_squared_error(target_batch_unnormalized, pred_batch_unnormalized[:, :, :2]) * mask
            num_time_steps_per_track = tf.reduce_sum(mask, axis=-1)
            batch_loss_per_track = tf.reduce_sum(batch_loss, axis=-1) / num_time_steps_per_track

            batch_mae = tf.keras.losses.mean_absolute_error(target_batch_unnormalized, pred_batch_unnormalized[:, :, :2]) * mask
            batch_mae_per_track = tf.reduce_sum(batch_mae, axis=-1) / num_time_steps_per_track

            # -----------

            target_x = target_batch[:, :, 0]
            target_y = target_batch[:, :, 1]

            pos_pred_x = batch_predictions[:, :, 0]
            pos_pred_y = batch_predictions[:, :, 1]

            log_var_pred_x = batch_predictions[:, :, 2]
            log_var_pred_y = batch_predictions[:, :, 3]

            pos_loss_x = (target_x - pos_pred_x) ** 2

            bnn_loss_x = 0.5 * K.exp(-log_var_pred_x) * pos_loss_x + 0.5 * log_var_pred_x

            pos_loss_y = (target_y - pos_pred_y) ** 2
            bnn_loss_y = 0.5 * K.exp(-log_var_pred_y) * pos_loss_y + 0.5 * log_var_pred_y

            neg_log_likelihood = mask * (bnn_loss_x + bnn_loss_y)

            neg_log_likelihood = K.sum(neg_log_likelihood) / K.sum(mask) + tf.add_n(self.rnn_model.losses)

            mses = np.concatenate((mses, batch_loss_per_track.numpy().reshape([-1])))
            maes = np.concatenate((maes, batch_mae_per_track.numpy().reshape([-1])))
            nlls = np.concatenate((nlls, neg_log_likelihood.numpy().reshape([-1])))

        test_mae = np.mean(maes)
        test_mse = np.mean(mses)
        test_nll = np.mean(nlls)

        logging.info("Evaluate: MSE={}".format(test_mse))
        logging.info("Evaluate: MAE={}".format(test_mae))
        logging.info("Evaluate: NLL={}".format(test_nll))

        plt.rc('grid', linestyle=":")
        fig1, ax1 = plt.subplots()

        if self.data_source.normalization_constant > 2.0:
            # ToDo: make this an argument
            ax1.yaxis.grid(True)
            ax1.set_ylim([0, 4.0])

        name = '{:05d}epoch-NextStep-RNN ({})'.format(epoch, self.model_hash)
        ax1.set_title(name)
        prop = dict(linewidth=2.5)
        ax1.boxplot(maes * self.data_source.normalization_constant, showfliers=False, boxprops=prop, whiskerprops=prop,
                    medianprops=prop, capprops=prop)
        plt.savefig(os.path.join(self.global_config['diagrams_path'], name + '.pdf'))
        plt.clf()

        return test_mse, test_mae, test_nll

    def _evaluate_custom_loss_model(self, dataset_test, epoch):
        mses = np.array([])
        maes = np.array([])
        custom_losses = np.array([])

        mask_value = K.variable(np.array([self.data_source.nan_value, self.data_source.nan_value]), dtype=tf.float64)
        normalization_factor = self.data_source.normalization_constant

        _ = self.rnn_model.reset_states()
        for input_batch, target_batch in dataset_test:
            if self.global_config['clear_state']:
                _ = self.rnn_model.reset_states()

            batch_predictions = self.rnn_model(input_batch)

            # Calculate the mask
            mask = K.all(K.equal(target_batch, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            target_batch_unnormalized = target_batch
            pred_batch_unnormalized = batch_predictions

            batch_loss = tf.keras.losses.mean_squared_error(target_batch_unnormalized, pred_batch_unnormalized[:, :, :2]) * mask
            num_time_steps_per_track = tf.reduce_sum(mask, axis=-1)
            batch_loss_per_track = tf.reduce_sum(batch_loss, axis=-1) / num_time_steps_per_track

            batch_mae = tf.keras.losses.mean_absolute_error(target_batch_unnormalized, pred_batch_unnormalized[:, :, :2]) * mask
            batch_mae_per_track = tf.reduce_sum(batch_mae, axis=-1) / num_time_steps_per_track

            # -----------

            target_x = target_batch[:, :, 0]
            target_y = target_batch[:, :, 1]

            pos_pred_x = batch_predictions[:, :, 0]
            pos_pred_y = batch_predictions[:, :, 1]

            log_var_pred_x = batch_predictions[:, :, 2]
            log_var_pred_y = batch_predictions[:, :, 3]

            pos_loss_x = (target_x - pos_pred_x) ** 2
            pos_loss_y = (target_y - pos_pred_y) ** 2

            var_loss_x = (pos_loss_x - K.exp(log_var_pred_x)) ** 2
            var_loss_y = (pos_loss_y - K.exp(log_var_pred_y)) ** 2

            custom_loss = mask * (pos_loss_x + pos_loss_y + var_loss_x + var_loss_y)

            custom_loss = K.sum(custom_loss) / K.sum(mask) + tf.add_n(self.rnn_model.losses)

            mses = np.concatenate((mses, batch_loss_per_track.numpy().reshape([-1])))
            maes = np.concatenate((maes, batch_mae_per_track.numpy().reshape([-1])))
            custom_losses = np.concatenate((custom_losses, custom_loss.numpy().reshape([-1])))

        test_mae = np.mean(maes)
        test_mse = np.mean(mses)
        test_custom_loss = np.mean(custom_losses)

        logging.info("Evaluate: MSE={}".format(test_mse))
        logging.info("Evaluate: MAE={}".format(test_mae))
        logging.info("Evaluate: custom_loss={}".format(test_custom_loss))

        plt.rc('grid', linestyle=":")
        fig1, ax1 = plt.subplots()

        if self.data_source.normalization_constant > 2.0:
            # ToDo: make this an argument
            ax1.yaxis.grid(True)
            ax1.set_ylim([0, 4.0])

        name = '{:05d}epoch-NextStep-RNN ({})'.format(epoch, self.model_hash)
        ax1.set_title(name)
        prop = dict(linewidth=2.5)
        ax1.boxplot(maes * self.data_source.normalization_constant, showfliers=False, boxprops=prop, whiskerprops=prop,
                    medianprops=prop, capprops=prop)
        plt.savefig(os.path.join(self.global_config['diagrams_path'], name + '.pdf'))
        plt.clf()

        return test_mse, test_mae, test_custom_loss

    def _evaluate_model(self, dataset_test, epoch):
        mses = np.array([])
        maes = np.array([])

        mask_value = K.variable(np.array([self.data_source.nan_value, self.data_source.nan_value]), dtype=tf.float64)
        normalization_factor = self.data_source.normalization_constant

        _ = self.rnn_model.reset_states()
        for input_batch, target in dataset_test:
            if self.global_config['clear_state']:
                _ = self.rnn_model.reset_states()

            # Calculate the mask
            mask = K.all(K.equal(target, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)
            np_mask = ~(K.all(K.equal(target, mask_value), axis=-1)).numpy()

            if self.global_config['mc_dropout'] and self.global_config['mc_samples'] > 1:
                samples = []
                k = self.global_config['mc_samples']
                for _ in range(k):
                    self.rnn_model.reset_states()
                    K2.set_learning_phase(1)
                    predic = self.rnn_model(input_batch, training=True)
                    samples.append(predic)

                samples = np.array(samples)
                sample_mean = np.sum(samples, axis=0) / float(k)
                # sample_variance = np.sum((samples - sample_mean) ** 2, axis=0) / float(k)

                predictions = sample_mean
                # variances = sample_variance
            else:
                predictions = self.rnn_model(input_batch)

            # se = squared error
            se_x = ((target[:, :, 0] - predictions[:, :, 0]) ** 2)
            se_y = ((target[:, :, 1] - predictions[:, :, 1]) ** 2)
            se = (((se_x + se_y) * mask) / K.sum(mask)).numpy()

            # ae = absolute error
            ae_x = ((target[:, :, 0] - predictions[:, :, 0]) ** 2) ** 0.5
            ae_y = ((target[:, :, 1] - predictions[:, :, 1]) ** 2) ** 0.5
            ae = (((ae_x + ae_y) * mask) / K.sum(mask)).numpy()

            # take average w.r.t. the number of unmasked entries
            # mse = K.sum(se) / K.sum(mask) + tf.add_n(self.rnn_model.losses)
            # mae = K.sum(ae) / K.sum(mask) + tf.add_n(self.rnn_model.losses)

            mses = np.concatenate((mses, se[np_mask].reshape([-1])))
            maes = np.concatenate((maes, ae[np_mask].reshape([-1])))

        test_mae = np.mean(maes)
        test_mse = np.mean(mses)

        logging.info("Evaluate: MSE={}".format(test_mse))
        logging.info("Evaluate: MAE={}".format(test_mae))

        plt.rc('grid', linestyle=":")
        fig1, ax1 = plt.subplots()

        if self.data_source.normalization_constant > 2.0:
            # ToDo: make this an argument
            ax1.yaxis.grid(True)
            ax1.set_ylim([0, 4.0])

        name = '{:05d}epoch-NextStep-RNN ({})'.format(epoch, self.model_hash)
        ax1.set_title(name)
        prop = dict(linewidth=2.5)
        ax1.boxplot(maes * self.data_source.normalization_constant, showfliers=False, boxprops=prop, whiskerprops=prop,
                    medianprops=prop, capprops=prop)
        plt.savefig(os.path.join(self.global_config['diagrams_path'], name + '.pdf'))
        plt.clf()

        return test_mse, test_mae

    def _evaluate_separation_model(self, dataset_test, epoch):
        prediction_maes = np.array([])
        spatial_errors = np.array([])
        time_errors = np.array([])

        mask_value = K.variable(np.array([self.data_source.nan_value, self.data_source.nan_value]), dtype=tf.float64)
        normalization_factor = self.global_config['CsvDataSet']['normalization_constant']
        time_normalization_constant = self.global_config['time_normalization_constant']

        hidden = self.rnn_model.reset_states()
        for input_batch, target_batch in dataset_test:
            if self.global_config['clear_state']:
                _ = self.rnn_model.reset_states()

            batch_predictions = self.rnn_model(input_batch)
            batch_predictions_np = batch_predictions.numpy()

            # Calculate the mask
            mask = K.all(K.equal(input_batch, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            batch_loss = tf.keras.losses.mean_absolute_error(target_batch[:, :, :2], batch_predictions[:, :, :2]) * mask * normalization_factor
            num_time_steps_per_track = tf.reduce_sum(mask, axis=-1)
            batch_loss_per_track = tf.reduce_sum(batch_loss, axis=-1) / num_time_steps_per_track

            # Spatial and temporal error

            # taken from the last timestep (this is correct due to masking which copies
            #   the last values until the end)
            spatial_diff = (batch_predictions[:, -1, 2:3] - target_batch[:, -1, 2:3]).numpy().flatten() * normalization_factor

            # temporal diff:
            #   example: label=17.3
            #    -> get prediction of timestep 17 -> could be 0.9
            #    => 0.9 - (17.3 - 17)
            temporal_diff = []
            for track_i in range(target_batch.shape[0]):
                track_input = input_batch[track_i]
                last_time_step = self.data_source.get_last_timestep_of_track(track_input) - 1
                target_sep_time = target_batch[track_i, 0, 3] - last_time_step / self.global_config[
                    'time_normalization_constant']
                pred_sep_time = batch_predictions[track_i, -1, 3]
                time_error = self.global_config['time_normalization_constant'] * (pred_sep_time - target_sep_time)
                temporal_diff.append(time_error)
            temporal_diff = np.array(temporal_diff)

            spatial_errors = np.concatenate((spatial_errors, spatial_diff))
            time_errors = np.concatenate((time_errors, temporal_diff))
            prediction_maes = np.concatenate((prediction_maes, batch_loss_per_track.numpy().reshape([-1])))

        test_pred_loss = np.mean(prediction_maes)
        test_spatial_loss = np.mean(spatial_errors)
        test_time_loss = np.mean(time_errors)
        test_loss = test_pred_loss + test_time_loss + test_spatial_loss

        logging.info("Evaluate: Mean Next Step Error = {}".format(test_pred_loss))
        logging.info("Evaluate: Mean Sep Spatial Error = {}".format(test_spatial_loss))
        logging.info("Evaluate: Mean Sep Time Error = {}".format(test_time_loss))
        logging.info("Evaluate: Sum of Error = {}".format(test_loss))

        #  [0, 4.0]
        self._box_plot(prediction_maes, None, 'Next-Step', epoch, 'MAE')
        # [-59, 59]
        self._box_plot(spatial_errors, None, 'Separation-Spatial', epoch, 'Spatial error')
        self._box_plot(100 * time_errors, None, 'Separation-Temporal', epoch, 'Temporal error [1/100 Frames]')

        return test_loss, test_pred_loss, test_spatial_loss, test_time_loss

    def _box_plot(self, errors, y_lim, name, epoch, ylabel):
        plt.rc('grid', linestyle=":")
        fig1, ax1 = plt.subplots()
        ax1.yaxis.grid(True)
        if y_lim is not None:
            ax1.set_ylim(y_lim)

        file_name = '{:05d}epoch-{} ({})'.format(epoch, name, self.model_hash)
        ax1.set_title(name)
        plt.ylabel(ylabel)
        prop = dict(linewidth=2.5)
        ax1.boxplot(errors, showfliers=False, boxprops=prop, whiskerprops=prop, medianprops=prop, capprops=prop)
        plt.savefig(os.path.join(self.global_config['diagrams_path'], file_name + '.pdf'))
        plt.clf()

    def _get_evaluation_data(self, dataset):
        """
        Returns a dict which contains all of the data necessary to calculate the calibration and correlation plots.

        Calc for every measurement:
           - measurement
           - prediction
           - prediction_variance

        :param dataset:
        :return:
        """
        data = {}

        if self.global_config['mc_dropout'] or self.global_config['kendall_loss'] or self.global_config['custom_variance_prediction']:
            data['measurement'] = []
            data['target'] = []
            data['prediction'] = []
            data['prediction_variance'] = []
            data['variance_area'] = []
            data['standardized_l2'] = []
            data['l2'] = []
            data['absolute_error'] = []
            data['squared_error'] = []
            data['time_step'] = []

            for (batch_n, (inp_batch, target_batch)) in enumerate(dataset):
                inp_batch = inp_batch.numpy()
                target_batch = target_batch.numpy()

                if self.global_config['mc_dropout'] and self.global_config['mc_samples'] > 1:
                    k = self.global_config['mc_samples']
                    K2.set_learning_phase(1)

                    samples = []

                    for t_k in range(k):
                        _ = self.rnn_model.reset_states()
                        predic = self.rnn_model(inp_batch, training=True)
                        samples.append(predic)

                    samples = np.array(samples)
                    sample_mean = np.sum(samples, axis=0) / float(k)
                    sample_variance = np.sum((samples - sample_mean) ** 2, axis=0) / float(k)

                    pos_predictions = sample_mean
                    var_predictions = sample_variance
                elif self.global_config['mc_dropout'] and self.global_config['mc_samples'] == 1:
                    _ = self.rnn_model.reset_states()
                    pos_predictions = self.rnn_model(inp_batch, training=False).numpy()
                    var_predictions = np.ones_like(pos_predictions) * 0.01
                else:
                    _ = self.rnn_model.reset_states()
                    predictions = self.rnn_model(inp_batch)

                    pos_predictions = predictions[:, :, :2].numpy()
                    var_predictions = K.exp(predictions[:, :, 2:]).numpy()

                # store data
                for track_idx in range(inp_batch.shape[0]):
                    seq_length = self.data_source.get_last_timestep_of_track(inp_batch[track_idx])

                    for time_step_i in range(seq_length):
                        data['prediction'] += [pos_predictions[track_idx, time_step_i, :]]
                        data['prediction_variance'] += [var_predictions[track_idx, time_step_i, :]]
                        # 4 * sigma_x * sigma_y
                        data['variance_area'] += [4 * np.sqrt(var_predictions[track_idx, time_step_i, 0]) *
                                                  np.sqrt(var_predictions[track_idx, time_step_i, 1])]
                        data['measurement'] += [inp_batch[track_idx, time_step_i, :]]
                        data['target'] += [target_batch[track_idx, time_step_i, :]]
                        data['standardized_l2'] += [np.sqrt(np.sum(
                            ((data['target'][-1] - data['prediction'][-1]) ** 2) / data['prediction_variance'][
                                -1]))]
                        data['absolute_error'] += [np.abs(data['prediction'][-1] - data['target'][-1])]
                        data['l2'] += [np.sqrt(np.sum(
                            (data['prediction'][-1] - data['target'][-1])**2
                        ))]
                        data['squared_error'] = [np.sum(
                            (data['prediction'][-1] - data['target'][-1])**2
                        )]
                        data['time_step'] += [time_step_i]

        data['prediction'] = np.array(data['prediction'])
        data['target'] = np.array(data['target'])
        data['prediction_variance'] = np.array(data['prediction_variance'])
        data['variance_area'] = np.array(data['variance_area'])
        data['measurement'] = np.array(data['measurement'])
        data['standardized_l2'] = np.array(data['standardized_l2'])
        data['l2'] = np.array(data['l2'])
        data['absolute_error'] = np.array(data['absolute_error'])
        data['time_step'] = np.array(data['time_step'])

        return data

    def plot_calibration(self, dataset, epoch=0):
        data = self._get_evaluation_data(dataset)
        N = data['time_step'].shape[0]

        def sigma_to_conf(sigmas):
            return erf(sigmas/np.sqrt(2))

        # evalute data
        stddevs = []
        cdf = []

        for stddev in np.arange(0.0, 10.0, 0.01):
            count_falls_into = np.count_nonzero(data['standardized_l2'] <= stddev)
            proportion = count_falls_into / N
            cdf.append(proportion)
            stddevs.append(stddev)

        stddevs = sigma_to_conf(np.array(stddevs))
        cdf = np.array(cdf)

        plt.scatter(stddevs, cdf)
        plt.title("Calibration plot (standardized euclidean distance)")
        plt.xlabel("Expected confidence level")
        plt.ylabel("Observed confidence level")
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'CDF_{}.pdf'.format(epoch)))
        plt.clf()



        #
        # stddevs = []
        # cdf = []
        #
        # for stddev in np.arange(0.0, 10.0, 0.01):
        #     mask = np.sqrt(data['prediction_variance'][:, 0]) < stddev
        #     mask_N = np.count_nonzero(mask)
        #     if mask_N == 0:
        #         continue
        #
        #     # vari_selected = data['prediction_variance'][:, 0][mask]
        #     pred_selected = data['prediction'][:, 0][mask]
        #     meas_selected = data['measurement'][:, 0][mask]
        #
        #     delta_x = np.abs(pred_selected - meas_selected)
        #     count_in = np.count_nonzero(delta_x <= stddev)
        #     proportion = count_in / mask_N
        #     cdf.append(proportion)
        #     stddevs.append(stddev)
        #
        # stddevs = sigma_to_conf(np.array(stddevs))
        # cdf = np.array(cdf)
        #
        # plt.scatter(stddevs, cdf)
        # plt.title("CDF2")
        # plt.savefig(os.path.join(self.global_config['diagrams_path'],
        #                                      'CDF2_{}.png'.format(epoch)))
        # plt.clf()

    def plot_correlation(self, dataset, epoch=0, data=None):
        if data is None:
            data = self._get_evaluation_data(dataset)
        N = data['time_step'].shape[0]

        self._plot_correlation_between(data['time_step'], data['l2'],
                                       '# Measurements per track', 'L2',
                                       epoch=epoch, violin=True)

        self._plot_correlation_between(data['time_step'], data['squared_error'],
                                       'Observed measurements per track', 'Squared Error',
                                       epoch=epoch, violin=True)

        self._plot_correlation_between(data['time_step'], data['standardized_l2'],
                                       'Observed Measurements per track', 'Standardized Euclidean L2',
                                       epoch=epoch, violin=True)

        self._plot_correlation_between(data['time_step'], data['variance_area'],
                                       'Observed Measurements per track', 'Variance rectangle',
                                       epoch=epoch, violin=True)

        self._plot_correlation_between(data['l2'], data['prediction_variance'][:, 0],
                                       'L2', 'Variance x',
                                       epoch=epoch)
        self._plot_correlation_between(data['l2'], data['prediction_variance'][:, 1],
                                       'L2', 'Variance y',
                                       epoch=epoch)
        self._plot_correlation_between(data['l2'], data['variance_area'],
                                       'L2', 'Variance rectangle',
                                       epoch=epoch)

    def _plot_correlation_between(self, x, y, x_name, y_name, epoch=0, violin=False):

        if violin:
            data = np.stack((x,y)).T
            violin_data = [data[data[:, 0] == x_val][:, 1] for x_val in set(x)]
            violin_parts = plt.violinplot(violin_data, widths=0.9,
                                          showmedians=False,
                                          showmeans=False,
                                          showextrema=False,
                                          points=1000)

            # set transparency of violin depending of count of points in bucket
            alpha_values = [len(violin_data_bucket) / float(data.shape[0]) for violin_data_bucket in violin_data]
            alpha_values /= np.max(alpha_values)

            for alpha, pc in zip(alpha_values, violin_parts['bodies']):
                pc.set_alpha(alpha)
        else:
            plt.scatter(x,y)

        try:
            r = pearsonr(x, y)[0]
        except ValueError as value_error:
            logging.error("Pearson value error")
            r = None

        # calc the trend line
        # z = np.polyfit(x, y, 1)
        # p = np.poly1d(z)
        # plt.plot(x, p(x), "b--")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        if r is not None:
            plt.title('Pearson r={:.2f} between {} and {}: (epoch={})'.format(r, x_name, y_name, epoch))
        else:
            plt.title('Pearson r=None between {} and {}: (epoch={})'.format(r, x_name, y_name, epoch))

        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'corr_{}_{}_{}.pdf'.format(x_name, y_name, epoch)))
        plt.clf()

    def plot_track_with_uncertainty(self, dataset, epoch=0, max_number_plots=10, tracks_per_plot=3,
                                    fit_scale_to_content=True, normed_plot=True, legend=False,
                                    point_size=0.25):
        if self.global_config['mc_dropout'] or self.global_config['kendall_loss'] or self.global_config['custom_variance_prediction']:

            for (batch_n, (inp_batch, target_batch)) in enumerate(dataset.take(1)):
                inp_batch = inp_batch.numpy()
                target_batch = target_batch.numpy()

                if self.global_config['mc_dropout'] and self.global_config['mc_samples'] > 1:
                    k = self.global_config['mc_samples']
                    K2.set_learning_phase(1)

                    samples = []

                    for t_k in range(k):
                        _ = self.rnn_model.reset_states()
                        predic = self.rnn_model(inp_batch, training=True)
                        samples.append(predic)

                    samples = np.array(samples)
                    sample_mean = np.sum(samples, axis=0) / float(k)
                    sample_variance = np.sum((samples - sample_mean) ** 2, axis=0) / float(k)

                    pos_predictions = sample_mean
                    var_predictions = sample_variance
                elif self.global_config['mc_dropout'] and self.global_config['mc_samples'] == 1:
                    _ = self.rnn_model.reset_states()
                    pos_predictions = self.rnn_model(inp_batch, training=False).numpy()
                    var_predictions = np.ones_like(pos_predictions) * 0.01
                else:
                    _ = self.rnn_model.reset_states()
                    predictions = self.rnn_model(inp_batch)

                    pos_predictions = predictions[:, :, :2].numpy()
                    var_predictions = K.exp(predictions[:, :, 2:]).numpy()

                stddev_predictions = np.sqrt(var_predictions)

                start_time_step = 0

                track_in_plot = 0

                for track_idx in range(min(max_number_plots*tracks_per_plot, inp_batch.shape[0]//tracks_per_plot)):
                    seq_length = self.data_source.get_last_timestep_of_track(inp_batch[track_idx])

                    axes = self.data_source.plot_track(inp_batch[track_idx], start=start_time_step,
                                                       fit_scale_to_content=fit_scale_to_content,
                                                       color='black', end=seq_length,
                                                       label="Input truth {}".format(track_idx),
                                                       normed_plot=normed_plot,
                                                       point_size=point_size,
                                                       legend=legend)
                    axes = self.data_source.plot_track(target_batch[track_idx], start=start_time_step,
                                                       fit_scale_to_content=fit_scale_to_content,
                                                       color='green', end=seq_length,
                                                       label="Output truth {}".format(track_idx),
                                                       normed_plot=normed_plot,
                                                       point_size=point_size,
                                                       legend=legend)
                    axes = self.data_source.plot_track(pos_predictions[track_idx], start=start_time_step,
                                                       fit_scale_to_content=fit_scale_to_content,
                                                       color='blue', end=seq_length,
                                                       label="Prediction {}".format(track_idx),
                                                       normed_plot=normed_plot,
                                                       point_size=point_size,
                                                       legend=legend)

                    ax = plt.subplot(111)

                    for time_step_i in range(seq_length):
                        ellipse = Ellipse(xy=pos_predictions[track_idx, time_step_i, :],
                                          width=2 * stddev_predictions[track_idx, time_step_i, 0],
                                          height=2 * stddev_predictions[track_idx, time_step_i, 1],
                                          angle=0)

                        ellipse.set_clip_box(ax.bbox)
                        ellipse.set_alpha(0.3)
                        ax.add_artist(ellipse)

                    plt.title('Tracks with Uncertainties')

                    track_in_plot += 1

                    if track_in_plot == tracks_per_plot:
                        track_in_plot = 0

                        legend_elements = [
                            Line2D([0], [0], marker='o', color='green', lw=2, label='True position',
                                   markersize=8, markerfacecolor='green'),
                            Line2D([0], [0], marker='o', color='blue', lw=2, label='Prediction mean',
                                   markersize=8, markerfacecolor='blue'),
                            Line2D([0], [0], marker='o', color='white', label='Prediction covariance',
                                   markerfacecolor='lightsteelblue', markersize=15)]

                        # Create the figure
                        plt.legend(handles=legend_elements, loc='upper right')

                        plt.savefig(os.path.join(self.global_config['diagrams_path'],
                                                 'Tracks_with_uncertainty_{}-{}.pdf'.format(epoch, track_idx)))
                        plt.clf()


