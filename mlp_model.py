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
    LR_Decay: 0.5
    LR_Decay_after_t: 150
    Batch size = 500
    Epochs = 3000
    Layers = [16, 16, 16]
    """

    __metaclass__ = Expert

    def __init__(self, name, model_path, is_next_step=True, mlp_config = {}):
        """Initialize the MLP model.

        Args:
            name (String):          Name of the model
            model_path (String):    Path to the model or path to save the model in
            is_next_step (Boolean): Do we train a tracking or an seperation prediction net
            mlp_config (dict):      Arguments for the mlp_model_factory
        """
        self.model_structure = mlp_config.get("model_structure")
        if "base_learning_rate" in mlp_config:
            self.base_learning_rate = mlp_config.get("base_learning_rate")
        else:
            self.base_learning_rate = 0.005
        self._label_dim = 2 if is_next_step else 4
        super().__init__(Expert_Type.MLP, name, model_path)

    def create_model(self, n_features):
        """Create a new MLP model.

        Args:
            n_features (int):       The number of features as input to the model   
        """
        self.mlp_model = mlp_model_factory(input_dim=n_features, output_dim=self._label_dim, **self.model_structure)
        self.input_dim = n_features
        logging.info(self.mlp_model.summary())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate)
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.train_step_fn = train_step_generator(self.mlp_model, self.optimizer, self.loss_object)

    def load_model(self):
        """Load a MLP model from its model path."""
        self.mlp_model = tf.keras.models.load_model(self.model_path)
        self.input_dim = self.mlp_model.input_shape[1]
        logging.info(self.mlp_model.summary())

    def train_batch(self, inp, target):
        """Train the MLP model on a batch of data.

        Args:
            inp (tf.Tensor): A batch of input tracks
            target (tf.Tensor): The prediction targets to the inputs

        Returns
            prediction (tf.Tensor): Predicted positions for training instances
        """
        return self.train_step_fn(inp, target)

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

def mlp_model_factory(input_dim=10, output_dim=2, layers=[16, 16, 16], activation='leakly_relu'):
    """Create a new keras MLP model

    Args:
        input_dim (int):    The number of features as input to the model
        output_dim (int):   The number of outputs of the model (usually 2 - [x, y])
    
    Returns: 
        the model
    """
    # define model
    model = tf.keras.Sequential()
    is_first = True
    # Add hidden layers
    for n_Neurons in layers:
        if is_first:
            model.add(tf.keras.layers.Dense(n_Neurons, kernel_initializer='he_normal', input_shape=(input_dim,)))
        else:
            model.add(tf.keras.layers.Dense(n_Neurons, kernel_initializer='he_normal'))
        if activation == 'leakly_relu':
            model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        else:
            logging.warning("Activation function {} not implemented yet :(".format(activation))

    # Add output layer
    model.add(tf.keras.layers.Dense(output_dim, activation='linear'))

    return model

def train_step_generator(model, optimizer, loss_object):
    """Build a function which returns a computational graph for tensorflow.

    This function can be called to train the given model with the given optimizer.

    Args:
        model:      model according to estimator api
        optimizer:  tf estimator
        nan_value:  e.g. 0

    Returns
        function which can be called to train the given model with the given optimizer
    """
    @tf.function
    def train_step(inp, target):
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            predictions = model(inp)

            loss = loss_object(target, predictions)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return predictions

    return train_step


# compile the model
#model.compile(optimizer='adam', loss='mse')
# fit the model
#model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# evaluate the model
#error = model.evaluate(X_test, y_test, verbose=0)
#print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))