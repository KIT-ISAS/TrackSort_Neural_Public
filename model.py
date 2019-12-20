import math

import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float64')


rnn_models = {
    'lstm': tf.keras.layers.LSTM,
    'rnn': tf.keras.layers.SimpleRNN,
    'gru': tf.keras.layers.GRU
}


def rnn_model_factory(
        num_units_first_rnn=16, num_units_second_rnn=16, num_units_third_rnn=0, num_units_fourth_rnn=0,
        num_units_first_dense=16, num_units_second_dense=0, num_units_third_dense=0, num_units_fourth_dense=0,
        rnn_model_name='lstm',
        use_batchnorm_on_dense=True,
        num_time_steps=35, batch_size=128, nan_value=0, input_dim=2, unroll=True, stateful=True):
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
                                stateful=stateful,
                                name='rnn-{}'.format(rnn_layer_count),
                                recurrent_initializer='glorot_uniform',
                                unroll=unroll))
            rnn_layer_count += 1

    # Add the dense layers
    for units_in_dense_layer in [num_units_first_dense, num_units_second_dense, num_units_third_dense,
                                 num_units_fourth_dense]:
        if units_in_dense_layer == 0:
            # as soon as one layer has no units, we don't create the layer, neither further ones
            break
        else:
            hash_ += "-dense[{}, leakyrelu]".format(units_in_dense_layer)
            model.add(tf.keras.layers.Dense(units_in_dense_layer))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            if use_batchnorm_on_dense:
                model.add(tf.keras.layers.BatchNormalization())
                hash_ += "-BatchNorm"

    # Always end with a dense layer with two outputs (x, y)
    model.add(tf.keras.layers.Dense(2))

    hash_ += "-dense[2]"

    return model, hash_


def tf_error(model, dataset, normalization_factor, squared=True, nan_value=0):
    """
    Build a function which returns a computational graph for tensorflow.

    :param model: keras model api
    :param dataset: iterable with shape [batch_size, time_steps, num_dims*2]
    :param normalization_factor: all floats get multiplied by this (usually belt_width)
    :param squared: if true, return MSE. Else: MAE
    :param nan_value: the value which was used for padding (usually 0)

    Example:
        >>> import data
        >>> keras_model = rnn_model_factory()
        >>> _, dataset_test = data.FakeDataSet().get_tf_data_sets_seq2seq_data()
        >>> calc_mse_test = model.tf_error(keras_model, dataset_test, normalization_factor, squared=True)
        >>> mse = calc_mse_test()
        >>> print("MSE: {}".format(mse))

    :return: method that constructs the graph. The result of the method can be called to calc the error
    """
    # the placeholder character used for padding
    mask_value = K.variable(np.array([nan_value, nan_value]), dtype=tf.float64)
    norm_factor = tf.constant(normalization_factor, dtype=tf.float64)

    @tf.function
    def f():
        # AutoGraph has special support for safely converting for loops when y is a tensor or tf.data.Dataset.
        # https://www.tensorflow.org/tutorials/customization/performance

        loss = tf.constant(0, dtype=tf.float64)
        step_counter = tf.constant(0, dtype=tf.float64)

        for input_batch, target_batch in dataset:
            # reset state
            hidden = model.reset_states()

            batch_predictions = model(input_batch)

            # Calculate the mask
            mask = K.all(K.equal(target_batch, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            # Revert the normalization
            # ToDo: Replace this hacky solution with same normalization in both directions
            #    with usage of x_max and y_max   with tf.scatter_nd_update(...)
            target_batch_unnormalized = target_batch * norm_factor
            pred_batch_unnormalized = batch_predictions * norm_factor

            if squared:
                batch_loss = tf.keras.losses.mean_squared_error(target_batch_unnormalized,
                                                                pred_batch_unnormalized) * mask
            else:
                batch_loss = tf.keras.losses.mean_absolute_error(target_batch_unnormalized,
                                                                 pred_batch_unnormalized) * mask

            # take average w.r.t. the number of unmasked entries
            loss += K.sum(batch_loss)
            step_counter += K.sum(mask)

        return loss/step_counter

    return f


def train_step_generator(model, optimizer, nan_value=0):
    """
    Build a function which returns a computational graph for tensorflow.
    This function can be called to train the given model with the given
    optimizer.

    :param model: model according to estimator api
    :param optimizer: tf estimator
    :param nan_value: e.g. 0

    Example:
        >>> import data
        >>> keras_model = rnn_model_factory()
        >>> dataset_train, _ = data.FakeDataSet().get_tf_data_sets_seq2seq_data()
        >>> train_step = model.train_step_generator(model, optimizer)
        >>>
        >>> step, batch_size = 0, 32
        >>> for epoch in range(100):
        >>>     for (batch_n, (inp, target)) in enumerate(dataset_train):
        >>>         # _ = keras_model.reset_states()
        >>>         loss = train_step(inp, target)
        >>>         step += batch_size
        >>>     tf.summary.scalar('loss', loss, step=step)

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

            # multiply categorical_crossentropy with the mask
            loss = tf.keras.losses.mean_squared_error(target, predictions) * mask

            # take average w.r.t. the number of unmasked entries
            loss = K.sum(loss) / K.sum(mask)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

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

        avg_loss = sum_loss/batch_counter

        return avg_loss, train_step_counter

    return train_epoch


class ModelManager(object):
    def __init__(self, maximum_number_of_tracks, batch_size, rnn_model, num_dims=2, dtype=np.float64):
        """
        Wrapper for an estimator containing layers with state (all layers inheriting from RNN)

        When used for tracking, we want to call the RNN with a full batch containing the current states of the tracks.
        The model manager makes available an api like the following:

        1. First you have to define the maximum number of tracks which shall be managed (if you have a batch size of 128
            and you want to use 140 tracks, then the model has to be run twice with different batches and states)
        2. For every track that you want to initialize, call: model_manager.allocate_track() which returns a track_id that
            you can use to reference the track. The state of the track is clean.
        3. Every track can use exactly one measurement to predict the next step.
           Call: model_manager.set_track_measurement(track_id, np.array([x, y], dtype=np.float64))
        4. Once you are done assigning measurements, you can run the prediction with:
            model_manager.predict(). This returns a list of predictions
        5. You can get the prediction for a track manually with model_manager.get_prediction(track_id)
        6. If you want to stop a track, call: model_manager.free(track_id)
        7. Resetting the full ModelManager: model_manager.free_all()

        :param maximum_number_of_tracks: total number of tracks which have to be handled simultaneously
        :param batch_size: batch size of the keras model
        :param rnn_model: keras model
        :param num_dims: how many dimensions has the input of one time step (if a position is given, then 2)
        :param dtype: the numpy datatype, either np.float64 or np.float32
        """
        self.maximum_number_of_tracks = maximum_number_of_tracks
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.dtype = dtype
        self.n_batches = math.ceil(maximum_number_of_tracks/batch_size)
        self.rnn_model = rnn_model

        # we start with cleaned states
        self.rnn_model.reset_states()

        # for every batch we store the clean state of a full model
        self.batch_states = [self._get_rnn_states() for _ in range(self.n_batches)]

        # store the occupied ids
        # the ids are in range(maximum_number_of_tracks) sharded across the batches
        self.used_state_ids = set()
        self.all_ids = set(range(maximum_number_of_tracks))

        # the measurements (only one time step)
        self.batch_measurements = [np.zeros([self.batch_size, 1, num_dims], dtype=self.dtype) for _ in range(self.n_batches)]

    def _pop_next_free_track_id(self):
        "Returns a new track id. If not available, then raises AssertionError"

        difference = self.all_ids - self.used_state_ids
        assert len(difference) > 0, "No track id left. Everything allocated."
        new_id = difference.pop()
        self.used_state_ids.add(new_id)
        return new_id

    def _get_rnn_states(self):
        "Returns all states of current batch sequential model"

        rnn_layer_states = []

        # get all layers in ascending order
        # ToDo: Replace this with while-true with break condition
        for i in range(1000):
            # Not asking for permission but handling the error is faster in python
            try:
                layer = self.rnn_model.get_layer(index=i)
            except:
                break

            # only store the state of the layer if it is a recurrent layer
            #   DenseLayers don't have a state
            if isinstance(layer, tf.keras.layers.RNN):
                rnn_layer_states.append([sub_state.numpy() for sub_state in layer.states])
                # print(rnn_layer_states)

        return rnn_layer_states

    def _get_batch_and_row_id(self, track_id):
        batch_id = track_id // self.batch_size
        row_id = track_id % self.batch_size
        return batch_id, row_id

    def _reset_state_of_track(self, track_id):
        batch_id, row_id = self._get_batch_and_row_id(track_id)
        batch_state = self.batch_states[batch_id]

        # for every layer in the batch state (e.g. lstm1 or lstm2)
        for layer_state_i in range(len(batch_state)):
            # for every sub_state in layer_state (e.g. cell memory and output of lstm)
            for sub_state_i in range(len(batch_state[layer_state_i])):
                # replace with zeros
                self.batch_states[batch_id][layer_state_i][sub_state_i][row_id] *= 0
                # sub_state[row_id] *= 0.0  # tf.zeros_like(sub_state[row_id], dtype=tf.float64)

    def allocate_track(self):
        new_id = self._pop_next_free_track_id()
        self._reset_state_of_track(new_id)
        return new_id

    def set_track_measurement(self, track_id, measurement):
        """
        Add the measurement for a track. Every track can only have one measurement.

        :param track_id:
        :param measurement: np.array([x,y], dtype=tf.float64)
        """
        batch_id, row_id = self._get_batch_and_row_id(track_id)
        self.batch_measurements[batch_id][row_id, 0, :] = measurement

    def _set_rnn_state(self, batch_state):
        rnn_layer_counter = 0
        for i in range(1000):
            try:
                layer = self.rnn_model.get_layer(index=i)
            except:
                break

            if isinstance(layer, tf.keras.layers.RNN):
                for sub_state_number, sub_state in enumerate(layer.states):
                    layer.states[sub_state_number].assign(tf.convert_to_tensor(batch_state[rnn_layer_counter][sub_state_number]))
                rnn_layer_counter += 1

    def predict(self):
        """
        Make the prediction for all tracks.
        If necessary the keras model predicts multiple times to use all data.

        :return: np.array([...], shape=[n_batches * batch_size, num_dims])
        """
        predictions = []
        for batch_i in range(self.n_batches):
            # set state for the batch
            self._set_rnn_state(self.batch_states[batch_i])
            # make prediction
            batch_predictions = self.rnn_model(self.batch_measurements[batch_i])
            predictions.append(batch_predictions)
            # store the state
            self.batch_states[batch_i] = self._get_rnn_states()

        return np.array(predictions).reshape([self.n_batches*self.batch_size, self.num_dims])

    def free(self, track_id):
        self.used_state_ids.remove(track_id)
        self.all_ids.add(track_id)

    def free_all(self):
        self.used_state_ids = set()
        self.all_ids = set(range(self.maximum_number_of_tracks))

    def get_all_measurements(self):
        return np.array(self.batch_measurements).reshape([self.n_batches * self.batch_size, self.num_dims])












