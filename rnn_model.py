import logging
import math
import os
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from expert import Expert, Expert_Type

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
        unroll=False, stateful=True):
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


def train_step_separation_prediction_generator(model, optimizer, nan_value=0):
    """Generate the train step function for the separation prediction.

    Args:
        model (tf.keras.Model):         The trainable tensorflow model
        optimizer (tf.keras.Optimizer): The optimizer (e.g. ADAM) 
        nan_value (any):                The padding value
    """

    @tf.function
    def train_step(inp, target, tracking_mask, separation_mask, train = True):
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            predictions = model(inp, training=train)
            tracking_loss = get_tracking_loss(predictions, target, tracking_mask)
            spatial_loss, temporal_loss = get_separation_loss(predictions, target, separation_mask)
            spatial_mae, temporal_mae = get_separation_mae(predictions, target, separation_mask)
            total_loss = tracking_loss + spatial_loss + temporal_loss

        if train:
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return predictions, spatial_loss, temporal_loss, spatial_mae, temporal_mae
    return train_step

def create_separation_masks(inp, mask_value=0, only_last_timestep_additional_loss=True):
    """Create two Keras masks for the separation prediction training.

    The first mask is for the tracking and has the shape [batch_size, track_length]. 
        Example track mask: [1, 1, 1, 1, 1, 0, 0, 0]
    The second mask is for the separation prediction and has the shape [batch_size, track_length]
        This mask depends on only_last_timestep_additional_loss.
        True: example mask: [0, 0, 0, 0, 1, 0, 0, 0]
        False:              [1, 1, 1, 1, 1, 0, 0, 0]

    Args:
        inp (tf.Tensor):    Input values in shape [batch_size, track_length, 2]
        mask_value (any):   The mask value to look for. Default is 0.
        only_last_timestep_additional_loss (Boolean): Changes the type of the separation tracking mask.

    Returns:
        tracking_mask, separation_mask
    """
    tracking_mask = 1 - tf.cast(tf.reduce_all(tf.equal(inp, mask_value), axis=-1), tf.int32)
    if only_last_timestep_additional_loss:
        mask_length = tf.reduce_sum(tracking_mask, axis=1) - 1
        separation_mask = tf.one_hot(indices=mask_length, depth=inp.shape.as_list()[1], off_value=0)
    else:
        separation_mask = tracking_mask
    return tf.cast(tracking_mask, tf.float64), tf.cast(separation_mask, tf.float64)

def get_tracking_loss(prediction, target, tracking_mask):
    """Calculate the tracking loss in the separation prediction training.

    tracking_loss = MSE([x,y] prediction<->target)

    Args:
        prediction (tf.Tensor): Predicted values [x, y, y_nozzle, dt_nozzle], shape: [batch_size, track_length, 4]
        target (tf.Tensor):     Target values [x, y, y_nozzle, dt_nozzle, y_velocity_nozzle], shape: [batch_size, track_length, 5]
        tracking_mask from create_separation_masks(...)

    Returns:
        The tracking loss
    """
    # Error on each track
    track_loss = tf.reduce_sum(tf.pow(target[:, :, :2]-prediction[:, :, :2], 2), axis=2) * tracking_mask
    # Total tracking error
    tracking_loss = tf.reduce_sum(track_loss)/tf.reduce_sum(tracking_mask)
    return tracking_loss

def get_separation_loss(prediction, target, separation_mask):
    """Calculate the spatial and temporal loss in the separation prediction training.

    temporal_loss = MSE([y_nozzle] prediction<->target)
    spatial_loss = MSE([dt_nozzle] prediction<->target)

    Args:
        prediction (tf.Tensor): Predicted values [x, y, y_nozzle, dt_nozzle], shape: [batch_size, track_length, 4]
        target (tf.Tensor):     Target values [x, y, y_nozzle, dt_nozzle, y_velocity_nozzle], shape: [batch_size, track_length, 5]
        separation_mask from create_separation_masks(...)

    Returns:
        spatial_loss, temporal_loss
    """
    # Spatial loss
    spatial_track_loss = tf.pow(target[:, :, 2]-prediction[:, :, 2], 2) * separation_mask
    spatial_loss = tf.reduce_sum(spatial_track_loss)/tf.reduce_sum(separation_mask)
     # Temporal loss
    temporal_track_loss = tf.pow(target[:, :, 3]-prediction[:, :, 3], 2) * separation_mask
    temporal_loss = tf.reduce_sum(temporal_track_loss)/tf.reduce_sum(separation_mask)
    return spatial_loss, temporal_loss

def get_separation_mae(prediction, target, separation_mask):
    """Calculate the spatial and temporal mae in the separation prediction training.

    temporal_mae = MAE([y_nozzle] prediction<->target)
    spatial_mae = MAE([dt_nozzle] prediction<->target)

    Args:
        prediction (tf.Tensor): Predicted values [x, y, y_nozzle, dt_nozzle], shape: [batch_size, track_length, 4]
        target (tf.Tensor):     Target values [x, y, y_nozzle, dt_nozzle, y_velocity_nozzle], shape: [batch_size, track_length, 5]
        separation_mask from create_separation_masks(...)

    Returns:
        spatial_mae, temporal_mae
    """
    # Spatial mae
    spatial_track_mae = tf.abs(target[:, :, 2]-prediction[:, :, 2]) * separation_mask
    spatial_mae = tf.reduce_sum(spatial_track_mae)/tf.reduce_sum(separation_mask)
     # Temporal mae
    temporal_track_mae = tf.abs(target[:, :, 3]-prediction[:, :, 3]) * separation_mask
    temporal_mae = tf.reduce_sum(temporal_track_mae)/tf.reduce_sum(separation_mask)
    return spatial_mae, temporal_mae


def train_step_generator(model, optimizer, loss_object, nan_value=0):
    """Build a function which returns a computational graph for tensorflow.

    This function can be called to train the given model with the given optimizer.

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
            predictions = model(inp) # TODO: MUSS HIER NICHT , training=True STEHEN?

            mask = K.all(K.equal(inp, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            loss = loss_object(target, predictions, sample_weight = mask)
            #mae = tf.keras.losses.mean_absolute_error(target, predictions, sample_weight = mask)

            # take average w.r.t. the number of unmasked entries
            #mse = K.sum(mse) / K.sum(mask)
            #mae = K.sum(mae) / K.sum(mask)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return predictions

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


class RNN_Model(Expert):

    __metaclass__ = Expert

    def __init__(self, is_next_step, name, model_path, rnn_config = {}):
        self.model_structure = rnn_config.get("model_structure")
        self.clear_state = rnn_config.get("clear_state")
        self.base_learning_rate = rnn_config.get("base_learning_rate") if "base_learning_rate" in rnn_config else 0.005
        self.decay_steps = rnn_config.get("decay_steps") if "decay_steps" in rnn_config else 200
        self.decay_rate = rnn_config.get("decay_rate") if "decay_rate" in rnn_config else 0.96
        self._label_dim = 2 if is_next_step else 4
        self.is_next_step = is_next_step
        super().__init__(Expert_Type.RNN, name, model_path)

    def get_zero_state(self, batch_size):
        self.rnn_model.reset_states()
        return get_state(self.rnn_model)

    # expected to return list<vector<pair<float,float>>>, list<RNNStateTuple>
    def predict(self, current_input, state):
        current_input = np.expand_dims(current_input, axis=1)
        set_state(self.rnn_model, state)
        prediction = self.rnn_model(current_input)

        # ToDo: use the separation predictions! Currently I just drop them.
        prediction = np.copy(prediction.numpy()[:, :, :2])

        prediction = np.squeeze(prediction)
        new_state = get_state(self.rnn_model)
        return prediction, new_state

    def create_model(self, batch_size, num_time_steps):
        """Create a new RNN model.

        Args:
            batch_size (int):            The batch size of the data
            num_time_steps (int):        The number of timesteps in the longest track
            time_normalization (double): The normalization constant for the separation time. Only needed for separation prediction.
        """
        self.rnn_model, self.model_hash = rnn_model_factory(batch_size=batch_size, num_time_steps=num_time_steps, 
                                                           output_dim=self._label_dim, **self.model_structure)
        logging.info(self.rnn_model.summary())
        self.setup_model()

    def load_model(self):
        """Load a RNN model from its model path."""
        self.rnn_model = tf.keras.models.load_model(self.model_path)
        logging.info(self.rnn_model.summary())
        self.setup_model()

    def setup_model(self):
        """Setup the model.

        Call this function from create_model and load_model.

        Define:
            * learning rate schedule
            * optimizer
            * loss function
            * train_step_fn
        """
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.base_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule)
        self.loss_object = tf.keras.losses.MeanSquaredError()
        if self.is_next_step:
            self.train_step_fn = train_step_generator(self.rnn_model, self.optimizer, self.loss_object)
        else:
            self.train_step_fn = train_step_separation_prediction_generator(model = self.rnn_model, optimizer = self.optimizer)
        self.rnn_model.reset_states()

    def train_batch(self, inp, target):
        """Train the rnn model on a batch of data.

        Args:
            inp (tf.Tensor): A batch of input tracks
            target (tf.Tensor): The prediction targets to the inputs

        Returns
            prediction (tf.Tensor): Predicted positions for training instances
        """
        if self.clear_state:
            self.rnn_model.reset_states()
        return self.train_step_fn(inp, target)

    def train_batch_separation_prediction(self, inp, target, tracking_mask, separation_mask):
        """Train the rnn model on a batch of data.

        Args:
            inp (tf.Tensor): A batch of input tracks
            target (tf.Tensor): The prediction targets to the inputs
            tracking_mask (tf.Tensor): Mask the valid time steps for tracking
            separation_mask (tf.Tensor): Mask the valid time step(s) for the separation prediction

        Returns
            prediction (tf.Tensor): Predicted positions for training instances
        """
        if self.clear_state:
            self.rnn_model.reset_states()
        return self.train_step_fn(inp, target, tracking_mask, separation_mask)

    def test_batch_separation_prediction(self, inp, target, tracking_mask, separation_mask):
        """Test the rnn model on a batch of data.

        Args:
            inp (tf.Tensor): A batch of input tracks
            target (tf.Tensor): The prediction targets to the inputs
            tracking_mask (tf.Tensor): Mask the valid time steps for tracking
            separation_mask (tf.Tensor): Mask the valid time step(s) for the separation prediction

        Returns
            prediction (tf.Tensor): Predicted positions for training instances
        """
        if self.clear_state:
            self.rnn_model.reset_states()
        return self.train_step_fn(inp, target, tracking_mask, separation_mask, train=False)

    def predict_batch(self, inp):
        """Predict a batch of input data."""
        if self.clear_state:
            self.rnn_model.reset_states()
        return self.rnn_model(inp)

    def save_model(self):
        """Save the model to its model path."""
        folder_path = os.path.dirname(self.model_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.rnn_model.save(self.model_path)

    def change_learning_rate(self, lr_change=1):
        """Change the learning rate of the model optimizer.

        This can be used to lower the learning rate after n time steps to increase the accuracy.
        The change is implemented multiplicative. Set lr_change > 1 to increase and < 1 to decrease the lr.

        Args:
            lr_change (double): Change in learning rate (factorial)
        """
        old_lr = K.get_value(self.optimizer.lr)
        new_lr = old_lr * lr_change
        logging.info("Reducing learning rate from {} to {}.".format(old_lr, new_lr))
        K.set_value(self.optimizer.lr, new_lr)

    def get_separation_errors(self, seq2seq_target, prediction, tracking_mask, separation_mask):
        """Calculate spatial and temporal errors.

        Args:
            seq2seq_target (tf.Tensor): Target      [x, y, y_nozzle, dt_nozzle], shape: [batch_size, track_length, 4]
            prediction:                 Prediction  [x, y, y_nozzle, dt_nozzle], shape: [batch_size, track_length, 4]
            tracking_mask (tf.Tensor):  Mask for all valid tracking indices
            separation_mask (tf.Tensor):Mask for the indice(s) that are used to validate the separation prediction

        Returns:
            spatial_loss, temporal_loss, spatial_mae, temporal_mae
        """
        t_1 = time.time()
        spatial_loss, temporal_loss = get_separation_loss(prediction, seq2seq_target, separation_mask)
        t_2 = time.time()
        spatial_mae, temporal_mae = get_separation_mae(prediction, seq2seq_target, separation_mask)
        t_3 = time.time()
        print("Time for loss calculation = {} s; Time for mae calculation = {} s".format(t_2-t_1, t_3-t_2))
        return spatial_loss, temporal_loss, spatial_mae, temporal_mae
        

    ########## OLD FUNCTIONS
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
                if self.clear_state:
                    self.rnn_model.reset_states()
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
                    if self.clear_state:
                        self.rnn_model.reset_states()
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
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MSE.png'))
        plt.clf()

        # MAEs
        plt.plot(train_losses[:, 0], train_losses[:, 2], c='blue', label="Training MAE")
        plt.plot(test_losses[:, 0], test_losses[:, 2], c='red', label="Test MAE")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MAE.png'))
        plt.clf()

    def _evaluate_model(self, dataset_test, epoch):
        """Plot boxplot of MSE and MAE.

        TODO: Move evaluation functions into seperate class.
                This function should only return the predictions.
        """
        mses = np.array([])
        maes = np.array([])

        mask_value = K.variable(np.array([self.data_source.nan_value, self.data_source.nan_value]), dtype=tf.float64)
        normalization_factor = self.data_source.normalization_constant

        _ = self.rnn_model.reset_states()
        for input_batch, target_batch in dataset_test:
            if selfclear_state:
                self.rnn_model.reset_states()

            batch_predictions = self.rnn_model(input_batch)

            # Calculate the mask
            mask = K.all(K.equal(input_batch, mask_value), axis=-1)
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

        test_mae = np.mean(maes)
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
        plt.savefig(os.path.join(self.global_config['diagrams_path'], name + '.png'))
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
            if self.clear_state:
                self.rnn_model.reset_states()

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
        plt.savefig(os.path.join(self.global_config['diagrams_path'], file_name + '.png'))
        plt.clf()
