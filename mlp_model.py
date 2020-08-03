"""MLP model.

TODO:
"""

import os
import logging

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

from expert import Expert, Expert_Type

class MLP_Model(Expert):
    """MLP model to predict next positions

    The MLP model has a fix number of previous points as input.
    Therefore, this model can only predict after a certain timestep.

    The data structure is:
    [x_in1, x_in2, ..., x_inN, y_in1, y_in2, ..., y_inN]
    [x_out, y_out]

    Good model properties:
    Activation: leaky relu
    optimizer: ADAM
    Basislernrate: 0.005
    decay_steps=200,
    decay_rate=0.96,
    staircase=True
    Batch size = 500
    Epochs = 3000
    Layers = [16, 16, 16]
    """

    __metaclass__ = Expert

    def __init__(self, name, model_path, is_next_step=True, is_uncertainty_prediction=False, mlp_config = {}):
        """Initialize the MLP model.

        Args:
            name (String):          Name of the model
            model_path (String):    Path to the model or path to save the model in
            is_next_step (Boolean): Do we train a tracking or an seperation prediction net
            mlp_config (dict):      Arguments for the mlp_model_factory
        """
        self.is_next_step = is_next_step
        self.is_uncertainty_prediction = is_uncertainty_prediction
        self.model_structure = mlp_config.get("model_structure")
        self.base_learning_rate = mlp_config.get("base_learning_rate") if "base_learning_rate" in mlp_config else 0.005
        self.decay_steps = mlp_config.get("decay_steps") if "decay_steps" in mlp_config else 200
        self.decay_rate = mlp_config.get("decay_rate") if "decay_rate" in mlp_config else 0.96
        
        if is_uncertainty_prediction:
            self._label_dim = 4
        else:
            self._label_dim = 2
        super().__init__(Expert_Type.MLP, name, model_path)

    def create_model(self, n_features):
        """Create a new MLP model.

        Args:
            n_features (int):       The number of features as input to the model   
        """
        self.mlp_model = mlp_model_factory(input_dim=n_features, output_dim=self._label_dim, **self.model_structure)
        self.input_dim = n_features
        logging.info(self.mlp_model.summary())
        self.setup_model()
        
    def load_model(self):
        """Load a MLP model from its model path."""
        self.mlp_model = tf.keras.models.load_model(self.model_path)
        try:
            self.input_dim = self.mlp_model.input_shape[1]
        except:
            self.input_dim = 14
            logging.warning("Input size was unknown. Most likely you tried to import your custom model which did not define the input shape in the first layer. Shame on you.")
        self.setup_model()
        logging.info(self.mlp_model.summary())
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
            self.train_step_fn = train_step_generator(self.mlp_model, self.optimizer, self.loss_object)
        else:
            self.train_step_fn = train_step_generator_separation_prediction(self.mlp_model, self.optimizer, self.loss_object, self.is_uncertainty_prediction)

    def train_batch(self, inp, target):
        """Train the MLP model on a batch of data.

        Args:
            inp (tf.Tensor): A batch of input tracks
            target (tf.Tensor): The prediction targets to the inputs

        Returns
            prediction (tf.Tensor): Predicted positions for training instances
        """
        return self.train_step_fn(inp, target)

    def train_batch_separation_prediction(self, inp, target, mask):
        """Train the MLP model on a batch of data.

        Args:
            inp (tf.Tensor):    A batch of input tracks
            target (tf.Tensor): The prediction targets to the inputs
            mask (tf.Tensor):   Indicates which tracks are valid

        Returns
            prediction (tf.Tensor):     Predicted positions for training instances
            spacial_loss (tf.Tensor):   MSE of y_nozzle prediction
            temporal_loss (tf.Tensor):  MSE of dt_nozzle prediction
            spacial_mae, temporal_mae:  MAE --""--
        """
        return self.train_step_fn(inp, target, mask, training=True)

    def test_batch_separation_prediction(self, inp, target, mask):
        """Test the MLP model on a batch of data.

        Args:
            inp (tf.Tensor):    A batch of input tracks
            target (tf.Tensor): The prediction targets to the inputs
            mask (tf.Tensor):   Indicates which tracks are valid

        Returns
            prediction (tf.Tensor):     Predicted positions for training instances
            spacial_loss (tf.Tensor):   MSE of y_nozzle prediction
            temporal_loss (tf.Tensor):  MSE of dt_nozzle prediction
            spacial_mae, temporal_mae:  MAE --""--
        """
        prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae = self.train_step_fn(inp, target, mask, training=False)
        prediction = self.correct_separation_prediction(np.array(prediction), mask)
        return prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae
 
    def correct_separation_prediction(self, prediction, mask):
        """Correct the uncertainty prediction of the expert with the ENCE calibration.

        Args:
            prediction (np.array): Predicted positions for training instances, shape = [n_tracks, 4]
                                    y_nozzle, dt_nozzle, sigma_y_nozzle, sigma_dt_nozzle
            mask (tf.Tensor):   Indicates which tracks are valid

        Returns:
            prediction
        """
        for track in range(prediction.shape[0]):
            if mask[track] == 0:
                std_y = np.sqrt(np.exp(prediction[track, 2]))
                # spatial correction
                corrected_std_y = self.calibration_separation_regression_var_spatial[0] * std_y + self.calibration_separation_regression_var_spatial[1]
                prediction[track, 2] = np.log(corrected_std_y**2)

                std_t = np.sqrt(np.exp(prediction[track, 3]))
                # temporal correction
                corrected_std_t = self.calibration_separation_regression_var_temporal[0] * std_t + self.calibration_separation_regression_var_temporal[1]
                prediction[track, 3] = np.log(corrected_std_t**2)
        return prediction

    def predict_batch(self, inp):
        """Predict a batch of input data."""
        return self.predict(inp)

    def predict(self, inp):
        """Predict input data."""
        return self.mlp_model(inp)

    def save_model(self):
        """Save the model to its model path."""
        folder_path = os.path.dirname(self.model_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.mlp_model.save(self.model_path)

    def get_zero_state(self, batch_size):
        """Return batch of empty lists."""
        return np.zeros([batch_size, self.input_dim])

    def get_input_dim(self):
        """Return input dimension."""
        return self.input_dim

    def build_new_state(self, measurement):
        """Return a new state with initial measurement"""
        state = np.zeros([self.input_dim])
        state[0] = measurement[0]
        state[int(self.input_dim/2)-1] = measurement[1]
        return state

    def update_state(self, state, measurement):
        """Update a given state with a new measurement.
        
        The state format is 
        x_in0, x_in1, ..., x_inN, y_in0, y_in1, ..., y_inN

        So the updating with a new measurement is done by rotating the state to the left
        x_in1, x_in2, ..., y_in0, y_in1, y_in2, ..., y_inN
        And replacing the middle and last position
        x_in1, x_in2, ..., x_new, y_in1, y_in2, ..., y_new

        Args:
            state (np array):           dim: [1, input_dim]
            measurement (np array):     dim: [2]

        The given state is updated. No need to return.
        """
        state[:-1] = state[1:]
        state[int(self.input_dim/2)-1] = measurement[0]
        state[-1] = measurement[1]
        


"""Model creation and training functionality"""

def mlp_model_factory(input_dim=10, output_dim=2, layers=[16, 16, 16], activation='leakly_relu',
                      l1_regularization=0, l2_regularization=0):
    """Create a new keras MLP model

    Args:
        input_dim (int):            The number of features as input to the model
        output_dim (int):           The number of outputs of the model (usually 2 - [x, y])
        layers (list):              List of integers. Each entry results in a new Dense layer with n neurons.
        activation (String):        Activation function of all layers (except output layer)
        l1_regularization (double): L1 Regularization factor
        l2_regularization (double): L2 Regularization factor

    Returns: 
        the model
    """
    # Define regularization
    regularization = tf.keras.regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)
    # Define model
    model = tf.keras.Sequential()
    is_first = True
    # Add hidden layers
    for n_Neurons in layers:
        if is_first:
            model.add(tf.keras.layers.Dense(n_Neurons, kernel_initializer='he_normal', input_shape=(input_dim,), 
                kernel_regularizer=regularization, bias_regularizer=regularization))
        else:
            model.add(tf.keras.layers.Dense(n_Neurons, kernel_initializer='he_normal', 
                kernel_regularizer=regularization, bias_regularizer=regularization))
        if activation == 'leakly_relu':
            model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        else:
            logging.warning("Activation function {} not implemented yet :(".format(activation))

    # Add output layer
    model.add(tf.keras.layers.Dense(output_dim, activation='linear', 
                kernel_regularizer=regularization, bias_regularizer=regularization))
    return model

def train_step_generator(model, optimizer, loss_object):
    """Build a function which returns a computational graph for tensorflow.

    This function can be called to train the given model with the given optimizer.

    Args:
        model:      model according to estimator api
        optimizer:  tf estimator
        loss_object: Loss object like keras.losses.MeanSquaredError

    Returns
        function which can be called to train the given model with the given optimizer
    """
    @tf.function
    def train_step(inp, target):
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            predictions = model(inp, training=True)

            loss = loss_object(target, predictions)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return predictions

    return train_step

def train_step_generator_separation_prediction(model, optimizer, loss_object, is_uncertainty_prediction):
    """Build a function which returns a computational graph for tensorflow.

    This function can be called to train the given model with the given optimizer.

    Args:
        model: model according to estimator api
        optimizer: tf estimator
        is_uncertainty_prediction (Boolean): Also predict the uncertainty of the output.

    Returns:
        function which can be called to train the given model with the given optimizer
    """
    stop=0
    @tf.function
    def train_step(inp, target, mask, training=True):
        with tf.GradientTape() as tape:
            #target = K.cast(target, tf.float64)
            predictions = model(inp, training=training, mask=mask)
            if is_uncertainty_prediction:
                spatial_loss, temporal_loss = get_separation_loss_uncertainty(predictions, target, mask)
            else:
                spatial_loss, temporal_loss = get_separation_loss(predictions, target, mask)
            spatial_mae, temporal_mae = get_separation_mae(predictions, target, mask)
            loss = spatial_loss + temporal_loss + tf.add_n(model.losses)
        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return predictions, spatial_loss, temporal_loss, spatial_mae, temporal_mae

    return train_step

def get_separation_loss(prediction, target, mask):
    """Calculate the spatial and temporal loss in the separation prediction training.

    temporal_loss = MSE([y_nozzle] prediction<->target)
    spatial_loss = MSE([dt_nozzle] prediction<->target)

    Args:
        prediction (tf.Tensor): Predicted values [y_nozzle, dt_nozzle], shape: [batch_size, track_length, 2]
        target (tf.Tensor):     Target values [y_nozzle, dt_nozzle], shape: [batch_size, track_length, 2]
        mask (tf.Tensor):       Indicates which instances are valid

    Returns:
        spatial_loss, temporal_loss
    """
    # Spatial loss
    spatial_loss = tf.reduce_sum(tf.pow(target[:, 0]-prediction[:, 0], 2) * mask)/tf.reduce_sum(mask)
     # Temporal loss
    temporal_loss = tf.reduce_sum(tf.pow(target[:, 1]-prediction[:, 1], 2) * mask)/tf.reduce_sum(mask)
    return spatial_loss, temporal_loss

def get_separation_loss_uncertainty(prediction, target, mask):
    """Calculate the spatial and temporal loss in the separation prediction training.
    temporal_loss = MSE([y_nozzle] prediction<->target)
    spatial_loss = MSE([dt_nozzle] prediction<->target)
    Args:
        prediction (tf.Tensor): Predicted values [y_nozzle, dt_nozzle, sigma_y_nozzle, sigma_dt_nozzle], shape: [batch_size, track_length, 4]
        target (tf.Tensor):     Target values [y_nozzle, dt_nozzle], shape: [batch_size, track_length, 2]
        mask (tf.Tensor):       Indicates which instances are valid
    Returns:
        spatial_loss, temporal_loss
    """
    # Spatial loss
    spatial_loss = tf.reduce_mean(tf.boolean_mask(0.5 * tf.exp(-prediction[:,2]) * tf.pow(target[:, 0]-prediction[:, 0], 2) + \
                                  0.5 * prediction[:,2], mask))
     # Temporal loss
    temporal_loss = tf.reduce_mean(tf.boolean_mask(0.5 * tf.exp(-prediction[:,3]) * tf.pow(target[:, 1]-prediction[:, 1], 2) + \
                                  0.5 * prediction[:,3], mask))
    return spatial_loss, temporal_loss

def get_separation_mae(prediction, target, mask):
    """Calculate the spatial and temporal MAE in the separation prediction training.

    temporal_loss = MAE([y_nozzle] prediction<->target)
    spatial_loss = MAE([dt_nozzle] prediction<->target)

    Args:
        prediction (tf.Tensor): Predicted values [y_nozzle, dt_nozzle], shape: [batch_size, track_length, 2]
        target (tf.Tensor):     Target values [y_nozzle, dt_nozzle], shape: [batch_size, track_length, 2]
        mask (tf.Tensor):       Indicates which instances are valid

    Returns:
        spatial_mae, temporal_mae
    """
    # Spatial mae
    spatial_mae = tf.reduce_sum(tf.abs(target[:, 0]-prediction[:, 0]) * mask)/tf.reduce_sum(mask)
     # Temporal mae
    temporal_mae = tf.reduce_sum(tf.abs(target[:, 1]-prediction[:, 1]) * mask)/tf.reduce_sum(mask)
    return spatial_mae, temporal_mae