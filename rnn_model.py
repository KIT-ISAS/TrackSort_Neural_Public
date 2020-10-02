"""Recurrent Neural Network model.

Change log (Please insert your name here if you worked on this file)
    * Created by: Daniel Pollithy 
    * Complete rework by: Jakob Thumm (jakob.thumm@student.kit.edu)
    * Jakob Thumm 2.10.2020:    Completed documentation.
"""
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

# The available recurrent model types. Just pick lstm and be happy.
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
    Create a new tf model with the sequential API

    Note: The number of layers is fixed, because the HParams Board can't handle lists at the moment (Nov-2019)

    Args:
        num_units_first_rnn (int):          Number of rnn nodes in first layer
        num_units_second_rnn (int):         Number of rnn nodes in second layer  
        num_units_third_rnn (int):          Number of rnn nodes in third layer
        num_units_fourth_rnn (int):         Number of rnn nodes in fourth layer
        num_units_first_dense (int):        Number of dense nodes in first layer
        num_units_second_dense (int):       Number of dense nodes in second layer
        num_units_third_dense (int):        Number of dense nodes in third layer
        num_units_fourth_dense (int):       Number of dense nodes in fourth layer
        rnn_model_name (string):            Model name
        use_batchnorm_on_dense (Boolean):   Use batch normalization on dense layers
        num_time_steps (int):               Number of timesteps in each track
        batch_size (int):                   Batch size
        nan_value:                          Value to mask out non valid measurements in track (0.0)
        input_dim (int):                    Input dimension (Should be 2 for [x, y]-measurement)
        output_dim (int):                   Output dimension - 2 for tracking, 4 for sep. pre. multitask, 6 for sep. pre. multitask with uncertainty 
        unroll (Boolean):                   Use unroll for the separation prediction (Not recommended - Choose multitask)
        stateful (Boolean):                 If this is False, then the state of the rnn will be reset after every batch.
                                                We want to control this manually therefore the default is True.
    Returns: 
        the model and a hash string identifying the architecture uniquely
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


def train_step_separation_prediction_generator(model, optimizer, is_uncertainty_prediction = False, nan_value=0):
    """Generate the train step function for the separation prediction.

    Args:
        model (tf.keras.Model):         The trainable tensorflow model
        optimizer (tf.keras.Optimizer): The optimizer (e.g. ADAM) 
        nan_value (any):                The padding value

    Returns:
        The train step function for separation prediction
    """
    @tf.function
    def train_step(inp, target, tracking_mask, separation_mask, train = True):
        """The train step function for the separation prediction.

        Args:
            inp (tf.Tensor):                Input [x, y], shape = [n_tracks, track_length, 2]
            target (tf.Tensor):             Target [x, y, y_nozzle, dt_nozzle, y_velocity_nozzle], shape = [n_tracks, track_length, 5]
            tracking_mask (tf.Tensor):      Indicates valid tracking indices with a 1, else 0, shape = [n_tracks, track_length]
            separation_mask (tf.Tensor):    One entry per track is 1 - the point where the separation prediction should be, shape = [n_tracks, track_length]
            train (Boolean):                Training activated?
        
        Returns:
            predictions (tf.Tensor):    Tracking and separation predictions (with uncertainty), shape = [n_tracks, track_length, 4/(6)]
                                            [x, y, y_nozzle, dt_nozzle, (log(var_y), log(var_dt))]
            spatial_loss (tf.Tensor):   Spatial MSE loss
            temporal_loss (tf.Tensor):  Temporal MSE loss
            spatial_mae (tf.Tensor):    Spatial MAE loss
            temporal_mae (tf.Tensor):   Temporal MAE loss
        """
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            predictions = model(inp, training=train)
            tracking_loss = get_tracking_loss(predictions, target, tracking_mask)
            if is_uncertainty_prediction:
                spatial_loss, temporal_loss = get_separation_loss_uncertainty(predictions, target, separation_mask)
            else:
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

    NOT IN USE ANYMORE

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
        prediction (tf.Tensor):         Predicted values [x, y, ...], shape = [batch_size, track_length, 2+]
        target (tf.Tensor):             Target values [x, y, ...], shape = [batch_size, track_length, 2+]
        tracking_mask (tf.Tensor):      Indicates valid tracking indices with a 1, else 0, shape = [n_tracks, track_length]
            
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
        prediction (tf.Tensor):         Predicted values [x, y, y_nozzle, dt_nozzle], shape = [batch_size, track_length, 4]
        target (tf.Tensor):             Target values [x, y, y_nozzle, dt_nozzle, ...], shape = [batch_size, track_length, 4+]
        separation_mask (tf.Tensor):    One entry per track is 1 - the point where the separation prediction should be, shape = [n_tracks, track_length]

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

def get_separation_loss_uncertainty(prediction, target, separation_mask):
    """Calculate the spatial and temporal loss in the separation prediction training.

    Uses the negative log-likelihood loss as described in Jakob Thumms fantastic master's thesis.

    Args:
        prediction (tf.Tensor):         Predicted values [x, y, y_nozzle, dt_nozzle, s_y, s_t], shape = [batch_size, track_length, 6]
                                            s_y = log(sigma_y^2)
        target (tf.Tensor):             Target values [x, y, y_nozzle, dt_nozzle, ...], shape = [batch_size, track_length, 4+]
        separation_mask (tf.Tensor):    One entry per track is 1 - the point where the separation prediction should be, shape = [n_tracks, track_length]

    Returns:
        spatial_loss, temporal_loss
    """
     # Spatial loss
    spatial_loss = tf.reduce_mean(tf.boolean_mask(0.5 * tf.exp(-prediction[:, :, 4]) * tf.pow(target[:, :, 2]-prediction[:, :, 2], 2) + \
                                  0.5 * prediction[:, :, 4], separation_mask))
     # Temporal loss
    temporal_loss = tf.reduce_mean(tf.boolean_mask(0.5 * tf.exp(-prediction[:, :, 5]) * tf.pow(target[:, :, 3]-prediction[:, :, 3], 2) + \
                                  0.5 * prediction[:, :, 5], separation_mask))
    return spatial_loss, temporal_loss

def get_separation_mae(prediction, target, separation_mask):
    """Calculate the spatial and temporal mae in the separation prediction training.

    temporal_mae = MAE([y_nozzle] prediction<->target)
    spatial_mae = MAE([dt_nozzle] prediction<->target)

    Args:
        prediction (tf.Tensor):         Predicted values [x, y, y_nozzle, dt_nozzle], shape = [batch_size, track_length, 4]
        target (tf.Tensor):             Target values [x, y, y_nozzle, dt_nozzle, ...], shape = [batch_size, track_length, 4+]
        separation_mask (tf.Tensor):    One entry per track is 1 - the point where the separation prediction should be, shape = [n_tracks, track_length]

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
    """Generate the train step function for tracking.

    Args:
        model (tf.keras.Model):         The trainable tensorflow model
        optimizer (tf.keras.Optimizer): The optimizer (e.g. ADAM) 
        loss_object (tf.keras.losses):  The loss object (e.g. tf.keras.losses.MeanSquaredError)
        nan_value (any):                The padding value

    Returns:
        The train step function for tracking
    """
    # the placeholder character used for padding
    mask_value = K.variable(np.array([nan_value, nan_value]), dtype=tf.float64)
    
    @tf.function
    def train_step(inp, target, training=True):
        """Train step function for tracking.

        Args:
            inp (tf.Tensor):    Input, [x, y], shape = [n_tracks, track_length, 2]
            target (tf.Tensor): Target, [x, y], shape = [n_tracks, track_length, 2]
            training (Boolean): Is the training activated?
        """
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            predictions = model(inp, training=training)

            mask = K.all(K.equal(target, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            loss = loss_object(target, predictions, sample_weight = mask)
            #mae = tf.keras.losses.mean_absolute_error(target, predictions, sample_weight = mask)

            # take average w.r.t. the number of unmasked entries
            #mse = K.sum(mse) / K.sum(mask)
            #mae = K.sum(mae) / K.sum(mask)
        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return predictions

    return train_step

def set_state(rnn_model, batch_state):
    """Set the state of a given RNN model manually.

    Args:
        rnn_model (tf.model):       The RNN model
        batch_state (tf.Tensor):    A batch of states
    """
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
    """Get the current state of the RNN model."""
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
    """RNN model for tracking and separation prediction.


    Good model properties:
    Activation: leaky relu
    optimizer: ADAM
    Basislernrate: 0.005
    decay_steps=200,
    decay_rate=0.96,
    staircase=True
    Batch size = 128
    Epochs = 1000
    Layers = [64, 16]
    """

    __metaclass__ = Expert

    def __init__(self, is_next_step, name, model_path, is_uncertainty_prediction = False, rnn_config = {}):
        """Create a RNN object.

        Args:
            is_next_step (Boolean):              True = Tracking, False = Separation Prediction
            name (String):                       Model name
            model_path (String):                 Path to save the model to
            is_uncertainty_prediction (Boolean): Predict an uncertainty
            rnn_config (dict):                   Parameters for rnn model creation
        """
        self.model_structure = rnn_config.get("model_structure")
        self.clear_state = rnn_config.get("clear_state")
        self.base_learning_rate = rnn_config.get("base_learning_rate") if "base_learning_rate" in rnn_config else 0.005
        self.decay_steps = rnn_config.get("decay_steps") if "decay_steps" in rnn_config else 200
        self.decay_rate = rnn_config.get("decay_rate") if "decay_rate" in rnn_config else 0.96
        self.is_uncertainty_prediction = is_uncertainty_prediction
        self._label_dim = 2 if is_next_step else 4
        if is_uncertainty_prediction and not is_next_step:
            # TODO: Change this when handling tracking again
            self._label_dim += 2
        self.is_next_step = is_next_step
        super().__init__(Expert_Type.RNN, name, model_path)

    def get_zero_state(self, batch_size):
        """Return default state for RNN model."""
        self.rnn_model.reset_states()
        return get_state(self.rnn_model)

    def predict(self, current_input, state):
        """Predict a batch of inputs for MTT."""
        current_input = np.expand_dims(current_input, axis=1)
        # TEST BUGFIX
        current_input[current_input==0]=-1
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
        self.load_calibration()

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
            # TODO: Implement uncertainty prediction
            self.train_step_fn = train_step_generator(self.rnn_model, self.optimizer, self.loss_object)
        else:
            self.train_step_fn = train_step_separation_prediction_generator(model = self.rnn_model, 
                                                                            optimizer = self.optimizer, 
                                                                            is_uncertainty_prediction=self.is_uncertainty_prediction)
        self.rnn_model.reset_states()

    def train_batch(self, inp, target):
        """Train the rnn model on a batch of data.

        Args:
            inp (tf.Tensor):    A batch of input tracks
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
            inp (tf.Tensor):                A batch of input tracks
            target (tf.Tensor):             The prediction targets to the inputs
            tracking_mask (tf.Tensor):      Mask the valid time steps for tracking
            separation_mask (tf.Tensor):    Mask the valid time step(s) for the separation prediction

        Returns
            prediction (tf.Tensor): Predicted positions for training instances
        """
        if self.clear_state:
            self.rnn_model.reset_states()
        return self.train_step_fn(inp, target, tracking_mask, separation_mask)

    def test_batch_separation_prediction(self, inp, target, tracking_mask, separation_mask):
        """Test the rnn model on a batch of data.

        Args:
            inp (tf.Tensor):                A batch of input tracks
            target (tf.Tensor):             The prediction targets to the inputs
            tracking_mask (tf.Tensor):      Mask the valid time steps for tracking
            separation_mask (tf.Tensor):    Mask the valid time step(s) for the separation prediction

        Returns
            prediction (tf.Tensor): Predicted positions for training instances
        """
        if self.clear_state:
            self.rnn_model.reset_states()
        prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae = self.train_step_fn(inp, target, tracking_mask, separation_mask, train=False)
        if self.is_uncertainty_prediction:
            prediction = self.correct_separation_prediction(np.array(prediction), separation_mask)
        return prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae

    def correct_separation_prediction(self, prediction, separation_mask):
        """Correct the uncertainty prediction of the expert with the ENCE calibration.

        Args:
            separation_mask (np.array): Indicates where the separation prediction entries are (end_track)
            prediction (np.array):      Predictions with uncertainty, shape = [n_tracks, n_timesteps, 6]
                Tracking entries:
                    prediction[i, 0:end_track, 0:2] = [x_pred, y_pred]
                Separation prediction entries:
                    prediction[i, end_track, 2] = y_nozzle_pred    (Predicted y position at nozzle array)
                    prediction[i, end_track, 3] = dt_nozzle_pred   (Predicted time to nozzle array)
                    prediction[i, end_track, 4] = log(var_y)       (Predicted variance of spatial prediction)
                    prediction[i, end_track, 5] = log(var_t)       (Predicted variance of temporal prediction)

        Returns:
            prediction (np.array):  Corrected Predictions, shape = [n_tracks, n_timesteps, 6]
        """
        for track in range(prediction.shape[0]):
            sep_pos = np.where(separation_mask[track] == 1)
            std_y = np.sqrt(np.exp(prediction[track, sep_pos, 4]))
            # spatial correction
            corrected_std_y = self.calibration_separation_regression_var_spatial[0] * std_y + self.calibration_separation_regression_var_spatial[1]
            prediction[track, sep_pos, 4] = np.log(corrected_std_y**2)
            std_t = np.sqrt(np.exp(prediction[track, sep_pos, 5]))
            # temporal correction
            corrected_std_t = self.calibration_separation_regression_var_temporal[0] * std_t + self.calibration_separation_regression_var_temporal[1]
            prediction[track, sep_pos, 5] = np.log(corrected_std_t**2)
        return prediction

    def predict_batch(self, inp):
        """Predict a batch of input data."""
        if self.clear_state:
            self.rnn_model.reset_states()
        return self.rnn_model(inp, training=False)

    def save_model(self):
        """Save the model to its model path."""
        folder_path = os.path.dirname(self.model_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.rnn_model.save(self.model_path)

    def change_learning_rate(self, lr_change=1):
        """Change the learning rate of the model optimizer.

        NOT IN USE ANYMORE -- USE lr shedule instead.

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
            seq2seq_target (tf.Tensor):     Target      [x, y, y_nozzle, dt_nozzle], shape = [batch_size, track_length, 4]
            prediction:                     Prediction  [x, y, y_nozzle, dt_nozzle], shape = [batch_size, track_length, 4]
            tracking_mask (tf.Tensor):      Mask for all valid tracking indices
            separation_mask (tf.Tensor):    Mask for the indice(s) that are used to validate the separation prediction

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
