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

        # Set the input dimension based on the features
        input_dim = 0
        for feature in self.model_structure.get("features"):
            if feature=="pos":
                input_dim = input_dim + 2
            elif feature=="id":
                input_dim = input_dim + 1
            elif feature=="prev_pred_err":
                input_dim = input_dim + n_experts
            else:
                logging.warning("Feature {} is unknown. Won't include it in ME gating.".format(feature))
        # Check if there is at least one input to the network
        assert(input_dim > 0)
        self.input_dim = input_dim
        # Output dimensions
        self._label_dim = n_experts
        super().__init__(n_experts, "ME weighting", model_path)

    def create_model(self):
        """Create a new MLP ME model based on the model structure defined in the config file."""
        self.mlp_model = me_mlp_model_factory(input_dim=self.input_dim, output_dim=self._label_dim, **self.model_structure)
        
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
        self.create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate)
        train_step_fn = train_step_generator(self.mlp_model, optimizer)

        ## Create Dataset
        # Transform the data to the desired data format
        inputs_model, targets_model, predictions_model, mask_model = self.transform_data(input_data, target, predictions, masks, self.model_structure.get("features"), self.input_dim)
        # Remove masked values from dataset
        inputs_model = inputs_model[np.sum(mask_model, axis=1)!=0]
        targets_model = targets_model[np.sum(mask_model, axis=1)!=0]
        predictions_model = predictions_model[np.sum(mask_model, axis=1)!=0]
        mask_model = mask_model[np.sum(mask_model, axis=1)!=0]
        # Create dataset
        dataset_input = tf.data.Dataset.from_tensor_slices((inputs_model, predictions_model, mask_model))
        dataset_target = tf.data.Dataset.from_tensor_slices((targets_model))
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
            for ((train_inputs, train_expert_predictions, train_mask), train_targets) in train_iter:
                # Predict weights for experts
                weights, loss = train_step_fn(train_inputs, train_targets, train_expert_predictions, train_mask)
                masked_weights = mask_weights(weights, train_mask)
                # Evaluate mae
                mae = weighted_sum_mae_loss(masked_weights, train_expert_predictions, train_targets)
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
                for ((test_inputs, test_expert_predictions, test_mask), test_targets) in test_iter:
                    # Predict weights for experts
                    weights = self.mlp_model(test_inputs)
                    masked_weights = mask_weights(weights, test_mask)
                    loss = weighted_sum_mse_loss(masked_weights, test_expert_predictions, test_targets)
                    # Evaluate mae
                    mae = weighted_sum_mae_loss(masked_weights, test_expert_predictions, test_targets)
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

        
    def transform_data(self, input_data, target_data, predictions_data, masks, features, input_dim):
        """Transform the given data to match the input and output data of the MLP.

        Args:
            input_data (np.array):      The current input to the experts, shape: [n_tracks, track_length, 2]
            target_data (np.array):     All target values of the given dataset, shape: [n_tracks, track_length, 2]
            predictions_data (np.array):All predictions for all experts, shape: [n_experts, n_tracks, track_length, 2]
            masks (np.array):           Mask to mask total prediction, shape: [n_experts, n_tracks, track_length]
            features (list[String]):    Features to create. Possibilities:
                                            "pos":  x and y position of current measurement.
                                            "id":   Current track id - Number of measurements in track
                                            "prev_pred_err": Previous prediction error of all experts
            input_dim (int):            Input dimension for MLP (could be calculated from features)

        Returns: 
            inputs (np.array):      Inputs to the MLP based on features, shape: [n_tracks * track_length, input_dim]
            targets (np.array):     Target values, shape: [n_tracks * track_length, 2]
            predictions (np.array): Predictions of the experts as input to the MLP, shape: [n_tracks * track_length, n_experts, 2]
            mask (np.array):        Mask to mask total prediction, shape: [n_tracks * track_length]
        """
        n_tracks = input_data.shape[0]
        track_length = input_data.shape[1]
        output_size = n_tracks*track_length
        n_experts = predictions_data.shape[0]
        # Generate input
        inputs_out = np.zeros([output_size, input_dim])
        targets_out = np.zeros([output_size, 2])
        predictions_out = np.zeros([output_size, n_experts, 2])
        mask_out = np.zeros([output_size, n_experts])
        for i in range(n_tracks):
            input_pos = 0
            for feature in features:
                if feature=="pos":
                    inputs_out[i*track_length:(i+1)*track_length, input_pos:input_pos+2] = input_data[i]
                    input_pos += 2
                elif feature=="id":
                    inputs_out[i*track_length:(i+1)*track_length, input_pos] = np.arange(0, track_length)
                    input_pos += 1
                elif feature=="prev_pred_err":
                    for e in range(n_experts):
                        err = np.sum(np.power(target_data[i]-predictions_data[e,i,:,:], 2), axis=-1)
                        # Add a time lag to the error
                        err[1:]=err[:-1]
                        # Default value for first error position -> This is tricky.
                        err[0]=1
                        inputs_out[i*track_length:(i+1)*track_length, input_pos] = err
                        input_pos += 1
                else:
                    logging.warning("Feature {} is unknown. Won't include it in ME gating.".format(feature))
            
            targets_out[i*track_length:(i+1)*track_length] = target_data[i]
            predictions_out[i*track_length:(i+1)*track_length] = np.swapaxes(predictions_data[:,i,:,:],0, 1)
            mask_out[i*track_length:(i+1)*track_length] = np.swapaxes(masks[:,i,:],0, 1)
        return inputs_out, targets_out, predictions_out, mask_out

    def create_input_data(self, input_data, features, input_dim, track_predictions=None, track_target=None, track_ids=None):
        """Create input data from track format.

        Args:
            input_data (np.array):          The current input to the experts, shape: [n_tracks, track_length, 2]
                                             OR shape: [n_tracks, 2] if predicting a single time step
            features (list[String]):        Features to create. Possibilities:
                                                "pos":  x and y position of current measurement.
                                                "id":   Current track id - Number of measurements in track
                                                "prev_pred_err": Previous prediction error of all experts
            input_dim (int):                Input dimension to neural network
            track_predictions (np.array):   Optional track predictions for experts if you chose to use the prev_pred_err feature, shape: [n_experts, n_tracks, track_length, 2]
            track_target (np.array):        Optional track target if you chose to use the prev_pred_err feature, shape: [n_tracks, track_length, 2]
            track_ids (np.array):           Switch to multi-target tracking mode if this value is not None.
                                            In this mode, we only get one instance of a track. 
                                            Therefore the track length is 1 and the track_length dimension in the track_input is omitted.
                                            The track ids hold the position of the instance in its track.
                                            Shape: [n_tracks]
        Returns: 
            inputs (np.array):      Inputs to the MLP, shape: [n_tracks * track_length, 3]
        """
        n_tracks = input_data.shape[0]
        # In case of live multi-target tracking, the track length is always 1.
        track_length = input_data.shape[1] if track_ids is None else 1
        output_size = n_tracks*track_length
        inputs_out = np.zeros([output_size, input_dim])
        for i in range(n_tracks):
            input_pos = 0
            for feature in features:
                if feature=="pos":
                    inputs_out[i*track_length:(i+1)*track_length, input_pos:input_pos+2] = input_data[i]
                    input_pos += 2
                elif feature=="id":
                    inputs_out[i*track_length:(i+1)*track_length, input_pos] = np.arange(0, track_length) if track_ids is None else track_ids[i]
                    input_pos += 1
                elif feature=="prev_pred_err":
                    for e in range(track_predictions.shape[0]):
                        err = np.sum(np.power(track_target[i]-track_predictions[e,i,:,:], 2), axis=-1)
                        # Add a time lag to the error
                        err[1:]=err[:-1]
                        # Default value for first error position -> This is tricky.
                        err[0]=1
                        inputs_out[i*track_length:(i+1)*track_length, input_pos] = err
                        input_pos += 1
                else:
                    logging.warning("Feature {} is unknown. Won't include it in ME gating.".format(feature))
        return inputs_out

    def create_mask_data(self, mask):
        """Create masks from track format.

        Args:
            masks (np.array): Mask to mask total prediction, shape: [n_experts, n_tracks, track_length] 
                                OR shape: [n_experts, n_tracks] if predicting a single time step

        Returns: 
            mask (np.array): Mask to mask total prediction, shape: [n_tracks * track_length, n_experts]
        """
        n_tracks = mask.shape[1]
        # In case of live multi-target tracking, the track length is always 1.
        track_length = mask.shape[2] if mask.ndim==3 else 1
        output_size = n_tracks*track_length
        n_experts = mask.shape[0] 
        mask_out = np.zeros([output_size, n_experts])
        if mask.ndim==3:
            for i in range(n_tracks):
                mask_out[i*track_length:(i+1)*track_length] = np.swapaxes(mask[:,i,:],0, 1)
        else:
            mask_out = np.swapaxes(mask,0, 1)
        return mask_out

    def convert_weights_to_tracks(self, weights, track_length):
        """Create input data from track format.

        Asserts weights.shape[0] % track_length == 0

        Args:
            weights (np.array): The predicted weights, shape: [n_tracks * track_length, n_experts]

        Returns: 
            track_weights (np.array): The predicted weights in track format, shape: [n_experts, n_tracks, track_length]
        """
        assert(weights.shape[0] % track_length == 0)
        if track_length > 1:
            track_weights = np.reshape(weights, [int(weights.shape[0] / track_length), track_length, self.n_experts])
            track_weights = np.swapaxes(track_weights, 0, 2).swapaxes(1, 2)
        else:
            track_weights = np.swapaxes(weights, 0, 1)
        return track_weights

    def get_weights(self, batch_size):
        """Return a weight vector.
        
        The weights sum to 1.
        All Weights are > 0.

        Returns:
            np.array with weights of shape [n_experts, batch_size]
        """
        pass

    def get_masked_weights(self, mask, track_input, track_predictions=None, track_target=None, track_ids=None):
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
            mask (np.array): Mask array with shape [n_experts, n_tracks, track_length]
            track_input (np.array): Data input in track format, shape: [n_tracks, track_length, 2]
            track_predictions (np.array): Optional track predictions for experts if you chose to use the prev_pred_err feature
            track_target (np.array): Optional track target if you chose to use the prev_pred_err feature
            track_ids (np.array):   Switch to multi-target tracking mode if this value is not None.
                                    In this mode, we only get one instance of a track. 
                                    Therefore the track length is 1 and the track_length dimension in the track_input is omitted.
                                    The track ids hold the position of the instance in its track.
                                    Shape: [n_tracks]

        Returns:
            np.array with weights of shape mask.shape
        """
        net_input = self.create_input_data(input_data=track_input, 
                                           features=self.model_structure.get("features"),
                                           input_dim=self.input_dim, 
                                           track_predictions=track_predictions, 
                                           track_target=track_target, 
                                           track_ids=track_ids)
            
        weights = self.mlp_model(net_input)
        masks = self.create_mask_data(mask)
        masked_weights = mask_weights(weights, masks)
        masked_weights = masked_weights.numpy()
        track_length = track_input.shape[1] if track_ids is None else 1
        track_weights = self.convert_weights_to_tracks(masked_weights, track_length)
        return track_weights


"""Model creation and training functionality"""

def me_mlp_model_factory(input_dim, output_dim, prediciton_input_dim=2, features=["pos"], layers=[16, 16, 16], activation='leakly_relu'):
    """Create a new keras MLP model

    Args:
        input_dim (int):            The number of inputs to the model
        output_dim (int):           The number of outputs of the model = number of experts --> number of weights
        prediciton_input_dim (int): Should be 2 for x and y position.
        features (list):            List of String. Features for MLP. Possibilities:
                                        "pos":  x and y position of current measurement.
                                        "id":   Current track id - Number of measurements in track
                                        "prev_pred_err": TODO: Previous prediction error of all experts
        layers (list):              List of int. Number of nodes and layers
        activation (String):        Activation function for layers
    
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
    # Output layer = Dense layer with softmax activation
    weights = tf.keras.layers.Dense(output_dim, name='weights')(x)
    weights = tf.keras.layers.Softmax()(weights)
    model = tf.keras.Model(inputs=inputs, outputs=weights)
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
    def train_step(inp, target, expert_predictions, mask):
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            weights = model(inp, training=True) 
            masked_weights = mask_weights(weights, mask)
            loss = weighted_sum_mse_loss(masked_weights, expert_predictions, target)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return weights, loss

    return train_step

def mask_weights(weights, mask):
    """Mask the tf weights vector.

    A weight vector of form:
    0.4 0.2 0.4
    0.4 0.3 0.3
    ...

    With mask of form:
    True False True
    True True True
    ...

    Results in:
    0.5 0.0 0.5
    0.4 0.3 0.3
    ...

    Args:
        weights (tf.Tensor): The outputs of the MLP network, shape: [n_output, n_experts]
        mask (tf.Tensor):    The mask for all experts, shape: [n_output, n_experts]

    Returns:
        masked_weights (tf.Tensor): Same shape as weights
    """
    masked_weights = tf.multiply(weights, mask)
    masked_weights, _ = tf.linalg.normalize(masked_weights, ord=1, axis=1)
    return masked_weights

def weighted_sum_mse_loss(weights, expert_predictions, target):
    """Return MSE for weighted expert predictions."""
    return tf.reduce_sum(tf.pow(target-tf.einsum('ijk,ij->ik', expert_predictions, weights),2),axis=1)

def weighted_sum_mae_loss(weights, expert_predictions, target):
    """Return MAE for weighted expert predictions."""
    return tf.reduce_sum(tf.abs(target-tf.einsum('ijk,ij->ik', expert_predictions, weights)),axis=1)
