import logging
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K

tf.keras.backend.set_floatx('float64')

rnn_models = {
    'lstm': tf.keras.layers.LSTM,
    'rnn': tf.keras.layers.SimpleRNN,
    'gru': tf.keras.layers.GRU
}


def rnn_model_factory(
        num_units_first_rnn=1024, num_units_second_rnn=16, num_units_third_rnn=0, num_units_fourth_rnn=0,
        num_units_first_dense=0, num_units_second_dense=0, num_units_third_dense=0, num_units_fourth_dense=0,
        rnn_model_name='lstm',
        use_batchnorm_on_dense=True,
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

    # Always end with a dense layer with
    # - two outputs (x, y)
    # - or: three outputs (x, y, y_separation)
    # - or: four outputs (x, y, y_separation, t_separation)
    model.add(tf.keras.layers.Dense(output_dim))

    hash_ += "-dense[{}]".format(output_dim)

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

        return loss / step_counter

    return f


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

            spatial_mse = tf.keras.losses.mean_squared_error(target[:, start:end, 2:3], predictions[:, start:end, 2:3]) * mask
            spatial_mae = tf.keras.losses.mean_absolute_error(target[:, start:end, 2:3], predictions[:, start:end, 2:3]) * mask

            if only_last_timestep_additional_loss:
                spatial_mse = K.sum(spatial_mse * mask_last_step)
                spatial_mae = K.sum(spatial_mae * mask_last_step)
            else:
                spatial_mse = K.sum(spatial_mse) / K.sum(mask)
                spatial_mae = K.sum(spatial_mae) / K.sum(mask)

            # new_time_target is like a countdown
            new_time_target = (target[:, :, 3:4] - range_)

            temporal_mse = tf.keras.losses.mean_squared_error(new_time_target[:, start:end], predictions[:, start:end, 3:4]) * mask
            temporal_mae = tf.keras.losses.mean_squared_error(new_time_target[:, start:end], predictions[:, start:end, 3:4]) * mask

            if only_last_timestep_additional_loss:
                temporal_mse = K.sum(temporal_mse * mask_last_step)
                temporal_mae = K.sum(temporal_mae * mask_last_step)
            else:
                temporal_mse = K.sum(temporal_mse) / K.sum(mask)
                temporal_mae = K.sum(temporal_mae) / K.sum(mask)

            mse = pred_mse + spatial_mse + temporal_mse
            mae = pred_mae + spatial_mae + temporal_mae

        if apply_gradients:
            grads = tape.gradient(mse, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return mse, mae, pred_mse, pred_mae, spatial_mse, spatial_mae, temporal_mse, temporal_mae

    return train_step


def train_step_generator(model, optimizer, nan_value=0):
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

            mask = K.all(K.equal(inp, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            mse = tf.keras.losses.mean_squared_error(target, predictions) * mask
            mae = tf.keras.losses.mean_absolute_error(target, predictions) * mask

            # take average w.r.t. the number of unmasked entries
            mse = K.sum(mse) / K.sum(mask)
            mae = K.sum(mae) / K.sum(mask)

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

        self._label_dim = 4 if self.global_config['separation_prediction'] else 2

        if self.global_config['is_loaded']:
            self.rnn_model = tf.keras.models.load_model(self.global_config['model_path'])
            logging.info(self.rnn_model.summary())
            self.rnn_model.reset_states()
        else:
            if self.global_config['separation_prediction']:
                self.train_separation_prediction()
            else:
                self.train()

    def get_zero_state(self):
        self.rnn_model.reset_states()
        return get_state(self.rnn_model)

    # expected to return list<vector<pair<float,float>>>, list<RNNStateTuple>
    def predict(self, current_input, state):
        new_states = []
        predictions = []
        current_input = np.expand_dims(current_input, axis=1)
        set_state(self.rnn_model, state)
        prediction = self.rnn_model(current_input)
        prediction = np.squeeze(prediction)
        new_state = get_state(self.rnn_model)
        return prediction, new_state

    def train(self):
        self.rnn_model, self.model_hash = rnn_model_factory(batch_size=self.global_config['batch_size'],
                                                            num_time_steps=self.data_source.longest_track,
                                                            output_dim=self._label_dim,
                                                            **self.global_config['rnn_model_factory'])
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
            for (batch_n, (inp, target)) in enumerate(dataset_train):
                # Mini-Batches
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
                    or (epoch+1) == self.global_config['num_train_epochs']:
                logging.info(log_string)
                test_mse, test_mae = self._evaluate_model(dataset_test, epoch)
                test_losses.append([epoch, test_mse, test_mae * self.data_source.normalization_constant])
            else:
                logging.debug(log_string)

        self.rnn_model.save(self.global_config['model_path'])

        # Visualize loss curve
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # MSE
        plt.plot(train_losses[:, 0], train_losses[:, 1], c='blue', label="Training MSE")
        plt.plot(test_losses[:, 0], test_losses[:, 1], c='red', label="Test MSE")
        plt.legend(loc="upper right")
        plt.savefig(self.global_config['diagrams_path'] + 'MSE.png')
        plt.clf()

        # MAE
        plt.plot(train_losses[:, 0], train_losses[:, 2], c='blue', label="Training MAE (not normalized)")
        plt.plot(test_losses[:, 0], test_losses[:, 2], c='red', label="Test MAE (not normalized)")
        plt.legend(loc="upper right")
        plt.savefig(self.global_config['diagrams_path'] + 'MAE.png')
        plt.clf()

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
        logging.info(self.rnn_model.summary())

        optimizer = tf.keras.optimizers.Adam()
        train_step_fn = train_step_separation_prediction_generator(self.rnn_model, optimizer,
                                                                   batch_size=self.global_config['batch_size'],
                                                                   num_time_steps=num_time_steps,
                                                                   time_normalization=self.global_config['time_normalization_constant'],
                                                                   only_last_timestep_additional_loss=self.global_config['only_last_timestep_additional_loss'],
                                                                   apply_gradients=True
                                                                   )

        test_step_fn = train_step_separation_prediction_generator(self.rnn_model, optimizer,
                                                                   batch_size=self.global_config['batch_size'],
                                                                   num_time_steps=num_time_steps,
                                                                   time_normalization=self.global_config[
                                                                       'time_normalization_constant'],
                                                                   only_last_timestep_additional_loss=
                                                                   self.global_config['only_last_timestep_additional_loss'],
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

            for (batch_n, (inp, target)) in enumerate(dataset_train):
                # Mini-Batches
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

                for (batch_n, (inp, target)) in enumerate(dataset_test):
                    # Mini-Batches
                    _ = self.rnn_model.reset_states()
                    errors = test_step_fn(inp, target)
                    errors_in_one_batch.append(errors)

                error_list = np.mean(errors_in_one_batch, axis=0).tolist()
                test_losses.append([epoch] + error_list)
            else:
                logging.debug(log_string)

        self._evaluate_separation_model(dataset_test, epoch)

        # Store meta info
        self.rnn_model.save(self.global_config['model_path'])

        # Visualize loss curve
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # MSEs
        plt.plot(train_losses[:, 0], train_losses[:, 1], c='blue', label="Training MSE")
        plt.plot(test_losses[:, 0], test_losses[:, 1], c='red', label="Test MSE")
        plt.legend(loc="upper right")
        plt.savefig(self.global_config['diagrams_path'] + 'MSE.png')
        plt.clf()

        # MAEs
        plt.plot(train_losses[:, 0], train_losses[:, 2], c='blue', label="Training MAE")
        plt.plot(test_losses[:, 0], test_losses[:, 2], c='red', label="Test MAE")
        plt.legend(loc="upper right")
        plt.savefig(self.global_config['diagrams_path'] + 'MAE.png')
        plt.clf()

    def _evaluate_model(self, dataset_test, epoch):
        mses = np.array([])
        maes = np.array([])

        mask_value = K.variable(np.array([self.data_source.nan_value, self.data_source.nan_value]), dtype=tf.float64)
        normalization_factor = self.data_source.normalization_constant

        for input_batch, target_batch in dataset_test:
            # reset state
            _ = self.rnn_model.reset_states()

            batch_predictions = self.rnn_model(input_batch)

            # Calculate the mask
            mask = K.all(K.equal(target_batch, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            target_batch_unnormalized = target_batch
            pred_batch_unnormalized = batch_predictions

            batch_loss = tf.keras.losses.mean_squared_error(target_batch_unnormalized, pred_batch_unnormalized) * mask
            num_time_steps_per_track = tf.reduce_sum(mask, axis=-1)
            batch_loss_per_track = tf.reduce_sum(batch_loss, axis=-1) / num_time_steps_per_track

            batch_mae = tf.keras.losses.mean_absolute_error(target_batch_unnormalized, pred_batch_unnormalized) * mask
            batch_mae_per_track = tf.reduce_sum(batch_mae, axis=-1) / num_time_steps_per_track

            mses = np.concatenate((mses, batch_loss_per_track.numpy().reshape([-1])))
            maes = np.concatenate((maes, batch_mae_per_track.numpy().reshape([-1])))

        test_mae = np.mean(maes) * normalization_factor
        test_mse = np.mean(mses)

        logging.info("Evaluate: MSE={}".format(test_mse))
        logging.info("Evaluate: MAE={}".format(test_mae))

        plt.rc('grid', linestyle=":")
        fig1, ax1 = plt.subplots()
        ax1.yaxis.grid(True)
        ax1.set_ylim([0, 4.0])

        name = '{:05d}epoch-NextStep-RNN ({})'.format(epoch, self.model_hash)
        ax1.set_title(name)
        prop = dict(linewidth=2.5)
        ax1.boxplot(maes * self.data_source.normalization_constant, showfliers=False, boxprops=prop, whiskerprops=prop,
                    medianprops=prop, capprops=prop)
        plt.savefig(self.global_config['diagrams_path'] + name + '.png')
        plt.clf()

        return test_mse, test_mae

    def _evaluate_separation_model(self, dataset_test, epoch):
        prediction_maes = np.array([])
        spatial_errors = np.array([])
        time_errors = np.array([])

        mask_value = K.variable(np.array([self.data_source.nan_value, self.data_source.nan_value]), dtype=tf.float64)
        normalization_factor = self.data_source.normalization_constant

        for input_batch, target_batch in dataset_test:
            # reset state
            hidden = self.rnn_model.reset_states()

            batch_predictions = self.rnn_model(input_batch)
            batch_predictions_np = batch_predictions.numpy()

            # Calculate the mask
            mask = K.all(K.equal(target_batch, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            target_batch_unnormalized = target_batch * normalization_factor
            pred_batch_unnormalized = batch_predictions[:, :, :2] * normalization_factor

            batch_loss = tf.keras.losses.mean_absolute_error(target_batch_unnormalized[:, :, :2], pred_batch_unnormalized) * mask
            num_time_steps_per_track = tf.reduce_sum(mask, axis=-1)
            batch_loss_per_track = tf.reduce_sum(batch_loss, axis=-1) / num_time_steps_per_track

            # Spatial and temporal error

            # signed error
            # taken from the last timestep (this is correct due to masking which copies
            #   the last values until the end)
            batch_difference = (batch_predictions - target_batch).numpy()
            spatial_diff = (batch_difference[:, -1, 2] * self.data_source.normalization_constant).flatten()

            # temporal diff:
            #   example: label=17.3
            #    -> get prediction of timestep 17 -> could be 0.9
            #    => 0.9 - (17.3 - 17)
            temporal_diff = []
            for track_i in range(target_batch.shape[0]):
                track_input = input_batch[track_i]
                last_time_step = self.data_source.get_last_timestep_of_track(track_input) - 1
                target_sep_time = target_batch[track_i, 0, 3] - last_time_step / self.global_config['time_normalization_constant']
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

        self._box_plot(test_pred_loss, [0, 4.0], 'Next-Step', epoch, 'MAE [px]')
        self._box_plot(test_spatial_loss, [-59, 59], 'Separation-Spatial', epoch, 'Spatial error [px]')
        self._box_plot(100 * test_time_loss, None, 'Separation-Temporal', epoch, 'Temporal error [1/100 Frames]')

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
        plt.savefig(self.global_config['diagrams_path'] + file_name + '.png')
        plt.clf()

    def predict_final(self, states, y_targetline):
        raise NotImplementedError

    def train_final(self, x_data, y_data, pred_point):
        raise NotImplementedError




