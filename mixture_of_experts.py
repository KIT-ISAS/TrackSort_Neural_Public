"""Mixture of expert approach for gating structure.

TODO:
    * Right now the input training data is split into train and test data for the gating network.
        You may want to change this into passing the test data directly to the training function.
"""
import numpy as np
import logging
import tensorflow as tf
import os
import datetime

from tensorflow.keras import backend as K

from gating_network import GatingNetwork
from expert import Expert_Type

class MixtureOfExperts(GatingNetwork):
    """This gating network learns the weights of experts with a MLP approach.

    The inputs to the network are:
        Current x and y position
        Current idx of position in track

    The outputs of the networks are:
        One weight for each expert between 0 and 1. The weights sum to 1.

    Attributes:
        n_experts (int):    Number of experts
        weights (np.array): Weights based on covariance
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts, model_path, network_options = {}):
        """Initialize a covariance weighting ensemble gating network.
        
        Args:
            n_experts (int):    Number of experts
            mlp_config (dict):  Arguments for the mlp_model_factory
        """
        # Initialize with zero weights
        self.model_structure = network_options.get("model_structure")
        # Training parameters
        self.base_learning_rate = 0.005 if not "base_learning_rate" in network_options else network_options.get("base_learning_rate")
        self.batch_size = 1000 if not "batch_size" in network_options else network_options.get("batch_size")
        self.n_epochs = 1000 if not "n_epochs" in network_options else network_options.get("n_epochs")
        self.lr_decay_after_epochs = 100 if not "lr_decay_after_epochs" in network_options else network_options.get("lr_decay_after_epochs")
        self.lr_decay_factor = 0.5 if not "lr_decay_factor" in network_options else network_options.get("lr_decay_factor")
        self.evaluate_every_n_epochs = 20 if not "evaluate_every_n_epochs" in network_options else network_options.get("evaluate_every_n_epochs")

        self._label_dim = n_experts
        super().__init__(n_experts, "ME weighting", model_path)

    def create_model(self, n_features):
        """Create a new MLP ME model.

        Args:
            n_features (int):  The number of features as input to the model   
        """
        self.mlp_model = me_mlp_model_factory(input_dim=n_features, output_dim=self._label_dim, **self.model_structure)
        
        logging.info(self.mlp_model.summary())

    def save_model(self):
        """Save the model to its model path."""
        folder_path = os.path.dirname(self.model_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.mlp_model.save(self.model_path)
        
    def load_model(self):
        """Load a MLP model from its model path."""
        self.mlp_model = tf.keras.models.load_model(self.model_path)
        self.input_dim = self.mlp_model.input_shape[1]
        logging.info(self.mlp_model.summary())

    def train_network(self, target, predictions, masks, input_data, expert_types, **kwargs):
        """Train the mixture of expert network.
        
        Create the network architecture.
        Transform the data to fit the network architecture.

        Args:
            targets (np.array):     All target values of the given dataset, shape: [n_tracks, track_length, 2]
            predictions (np.array): All predictions for all experts, shape: [n_experts, n_tracks, track_length, 2]
            masks (np.array):       Masks for each expert, shape: [n_experts, n_tracks, track_length]
            expert_types (list):    List of expert types
        """
        ## Create Model
        self.create_model(3)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate)
        train_step_fn = train_step_generator(self.mlp_model, optimizer)

        ## Create Dataset
        # Find a non MLP mask
        normal_mask_pos = 0
        for i in range(len(expert_types)):
            if not expert_types[i] == Expert_Type.MLP:
                normal_mask_pos = i 
                break
        mask = masks[normal_mask_pos]
        # Transform the data to the desired data format
        inputs_out, targets_out, predictions_out, mask_out = self.transform_data(input_data, target, predictions, mask)
        # Remove masked values from dataset
        inputs_model = inputs_out[mask_out==1]
        targets_model = targets_out[mask_out==1]
        predictions_model = predictions_out[mask_out==1]
        # Create dataset
        dataset_input = tf.data.Dataset.from_tensor_slices((inputs_model, predictions_model))
        dataset_target = tf.data.Dataset.from_tensor_slices(targets_model)
        dataset = tf.data.Dataset.zip((dataset_input, dataset_target))
        # Seperate dataset in train and test
        train_size = int(0.9 * inputs_model.shape[0])
        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        # Batch both datasets
        train_dataset = train_dataset.batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)
        
        ## Values for training logging in tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '_train_me_gating' + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '_train_me_gating' + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_mae = tf.keras.metrics.Mean('train_mae', dtype=tf.float32)
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        test_mae = tf.keras.metrics.Mean('test_mae', dtype=tf.float32)

        ## Training
        for epoch in range(self.n_epochs):
            # Learning rate decay
            if (epoch + 1) % self.lr_decay_after_epochs == 0:
                old_lr = K.get_value(optimizer.lr)
                new_lr = old_lr * self.lr_decay_factor
                logging.info("Reducing learning rate from {} to {}.".format(old_lr, new_lr))
                K.set_value(optimizer.lr, new_lr)
            # Iterate over the batches of the train dataset.
            train_iter = iter(train_dataset)
            for ((train_inputs, train_expert_predictions), train_targets) in train_iter:
                # Predict weights for experts
                weights, loss = train_step_fn(train_inputs, train_targets, train_expert_predictions)
                # Evaluate mae
                mae = weighted_sum_mae_loss(weights, train_expert_predictions, train_targets)
                train_loss(loss); train_mae(mae)
            # Write training results
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('mae', train_mae.result(), step=epoch)
            # Console output
            template = 'Epoch {}, Train Loss: {}, Train MAE: {}'
            logging.info(template.format(epoch+1,
                            train_loss.result().numpy(),
                            train_mae.result().numpy()))
            # Reset metrics every epoch
            train_loss.reset_states(); train_mae.reset_states()

            # Run trained models on the test set every n epochs
            if (epoch + 1) % self.evaluate_every_n_epochs == 0 \
                    or (epoch + 1) == self.n_epochs:
                # Iterate over the batches of the test dataset.
                test_iter = iter(test_dataset)
                for ((test_inputs, test_expert_predictions), test_targets) in test_iter:
                    # Predict weights for experts
                    weights = self.mlp_model(test_inputs)
                    loss = weighted_sum_mse_loss(weights, test_expert_predictions, test_targets)
                    # Evaluate mae
                    mae = weighted_sum_mae_loss(weights, test_expert_predictions, test_targets)
                    test_loss(loss); test_mae(mae)
                # Write testing results
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss.result(), step=epoch)
                    tf.summary.scalar('mae', test_mae.result(), step=epoch)
                # Console output
                template = 'Epoch {}, Test Loss: {}, Test MAE: {}'
                logging.info(template.format(epoch+1,
                                test_loss.result().numpy(),
                                test_mae.result().numpy()))
                # Reset metrics every epoch
                test_loss.reset_states(); test_mae.reset_states()

        
    def transform_data(self, input_data, target_data, predictions_data, mask):
        """Transform the given data to match the input and output data of the MLP.

        Args:
            input_data (np.array):      The current input to the experts, shape: [n_tracks, track_length, 2]
            target_data (np.array):     All target values of the given dataset, shape: [n_tracks, track_length, 2]
            predictions_data (np.array):All predictions for all experts, shape: [n_experts, n_tracks, track_length, 2]
            mask (np.array):            Mask to mask total prediction, shape: [n_tracks, track_length]

        Returns: 
            inputs (np.array):      Inputs to the MLP, shape: [n_tracks * track_length, 3]
            targets (np.array):     Target values, shape: [n_tracks * track_length, 2]
            predictions (np.array): Predictions of the experts as input to the MLP, shape: [n_tracks * track_length, n_experts, 2]
            mask (np.array):        Mask to mask total prediction, shape: [n_tracks * track_length]
        """
        n_tracks = input_data.shape[0]
        track_length = input_data.shape[1]
        output_size = n_tracks*track_length
        n_experts = predictions_data.shape[0]
        # Generate input
        inputs_out = np.zeros([output_size, 3])
        targets_out = np.zeros([output_size, 2])
        predictions_out = np.zeros([output_size, n_experts, 2])
        mask_out = np.zeros([output_size])
        for i in range(n_tracks):
            inputs_out[i*track_length:(i+1)*track_length,0:2] = input_data[i]
            inputs_out[i*track_length:(i+1)*track_length,2] = np.arange(0, track_length)
            targets_out[i*track_length:(i+1)*track_length] = target_data[i]
            predictions_out[i*track_length:(i+1)*track_length] = np.swapaxes(predictions_data[:,i,:,:],0, 1)
            mask_out[i*track_length:(i+1)*track_length] = mask[i]
        return inputs_out, targets_out, predictions_out, mask_out

    def get_weights(self, batch_size):
        """Return a weight vector.
        
        The weights sum to 1.
        All Weights are > 0.

        Returns:
            np.array with weights of shape [n_experts, batch_size]
        """
        weights = np.repeat(np.expand_dims(self.weights, -1), batch_size, axis=-1)
        return weights

    def get_masked_weights(self, mask):
        """Return an equal weights vector for all non masked experts.
        
        The weights sum to 1.
        All Weights are > 0 if the expert is non masked at an instance.

        If the mask value at an instance is 0, the experts weight is 0.

        example mask arry:
        [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
        --> Expert 6 is only active at position 0.
        
        Args:
            mask (np.array): Mask array with shape [n_experts, batch_size]

        Returns:
            np.array with weights of shape mask.shape
        """
        assert(mask.shape[0] == self.n_experts)
        batch_weight = self.get_weights(mask.shape[1])
        batch_weight = np.repeat(np.expand_dims(batch_weight, -1), mask.shape[2], axis=-1)
        epsilon = 1e-30
        weights = np.multiply(mask, batch_weight) / (np.sum(np.multiply(mask, batch_weight), axis=0) + epsilon)
        return weights


"""Model creation and training functionality"""

def me_mlp_model_factory(input_dim=3, prediciton_input_dim=2, output_dim=3, layers=[16, 16, 16], activation='leakly_relu'):
    """Create a new keras MLP model

    Args:
        input_dim (int):    The number of features as input to the model
        output_dim (int):   The number of outputs of the model (usually 2 - [x, y])
    
    Returns: 
        the model
    """
    # The input layer
    inputs = tf.keras.Input(shape=(input_dim,), name='me_input')

    # Add hidden layers
    is_first = True
    c=0
    for n_Neurons in layers:
        # Add layer
        if is_first:
            x = tf.keras.layers.Dense(n_Neurons, kernel_initializer='he_normal', name='dense_{}'.format(c))(inputs)
            is_first=False
        else:
            x = tf.keras.layers.Dense(n_Neurons, kernel_initializer='he_normal', name='dense_{}'.format(c))(x)
        # Add activation function to layer
        if activation == 'leakly_relu':
            x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        else:
            logging.warning("Activation function {} not implemented yet :(".format(activation))
        c+=1

    weights = tf.keras.layers.Dense(output_dim, name='weights')(x)
    weights = tf.keras.layers.Softmax()(weights)
    model = tf.keras.Model(inputs=inputs, outputs=weights)
    """
    model.compile(loss=weighted_sum_mse_loss(expert_predictions=expert_predictions), 
                  optimizer='ADAM',
                  experimental_run_tf_function=False) #metrics=[weighted_sum_mae_loss(expert_predictions=expert_predictions)],
    """
    return model

def train_step_generator(model, optimizer):
    """Build a function which returns a computational graph for tensorflow.

    This function can be called to train the given model with the given optimizer.

    Args:
        model:      model according to estimator api
        optimizer:  tf estimator

    Returns
        function which can be called to train the given model with the given optimizer
    """
    @tf.function
    def train_step(inp, target, expert_predictions):
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            weights = model(inp, training=True) 
            loss = weighted_sum_mse_loss(weights, expert_predictions, target)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return weights, loss

    return train_step

def weighted_sum_mse_loss(weights, expert_predictions, target):
    """Return MSE for weighted expert predictions."""
    return tf.reduce_sum(tf.pow(target-tf.einsum('ijk,ij->ik', expert_predictions, weights),2),axis=1)

def weighted_sum_mae_loss(weights, expert_predictions, target):
    """Return MAE for weighted expert predictions."""
    return tf.reduce_sum(tf.abs(target-tf.einsum('ijk,ij->ik', expert_predictions, weights)),axis=1)
