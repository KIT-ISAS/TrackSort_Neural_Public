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

class MixtureOfExpertsSeparation(GatingNetwork):
    """This gating network learns the weights of experts with a MLP approach.

    The inputs to the network are:
        Last n x and y positions
        (Current idx of position in track) not implemented yet

    The outputs of the networks are:
        Two weights for each expert between 0 and 1. The weights sum to 1.
        One weight for the temporal prediction, one for spatial prediction.

    Attributes:
        n_experts (int):    Number of experts
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts, model_path, is_uncertainty_prediction = False, network_options = {}):
        """Initialize a mixture of experts gating network for separation prediction.
        
        Args:
            n_experts (int):        Number of experts
            model_path (String):    Path to save/load the ME model
            is_uncertainty_prediction (Boolean): Predict uncertainty of predictions. 
            network_options (dict): Model training options
        """
        # Initialize with zero weights
        self.model_structure = network_options.get("model_structure")
        # Training parameters
        self.base_learning_rate = 0.005 if not "base_learning_rate" in network_options else network_options.get("base_learning_rate")
        self.batch_size = 64 if not "batch_size" in network_options else network_options.get("batch_size")
        self.n_epochs = 1000 if not "n_epochs" in network_options else network_options.get("n_epochs")
        self.decay_steps = 200 if not "decay_steps" in network_options else network_options.get("decay_steps")
        self.decay_rate = 0.96 if not "decay_rate" in network_options else network_options.get("decay_rate")
        self.n_inp_points = 7 if not "n_inp_points" in network_options else network_options.get("n_inp_points")
        self.evaluate_every_n_epochs = 20 if not "evaluate_every_n_epochs" in network_options else network_options.get("evaluate_every_n_epochs")
        if "is_one_out_of_n_selector" in network_options and network_options["is_one_out_of_n_selector"]:
            logging.warning("1 out of n selector does not work currently. What you would need to do is change the network structure. The last layer would need to be a categorical output layer.")
        self.is_one_out_of_n_selector = False
        #self.is_one_out_of_n_selector = False if not "is_one_out_of_n_selector" in network_options else network_options["is_one_out_of_n_selector"]
        self.is_uncertainty_prediction = is_uncertainty_prediction
        self.direct_uncertainty_output = False if not "direct_uncertainty_output" in network_options else network_options.get("direct_uncertainty_output")
        # Set the input dimension based on the features
        self.use_uncertainty_prediction_as_input = False
        input_dim = 0
        for feature in self.model_structure.get("features"):
            if feature=="pos":
                input_dim += 2 * self.n_inp_points
            elif feature=="id":
                logging.warning("Feature id is not implemented yet.")
            elif feature=="uncertainty_prediction":
                if is_uncertainty_prediction:
                    input_dim += 2*n_experts
                    self.use_uncertainty_prediction_as_input = True
                else:
                    logging.warning("Feature uncertainty prediction was chosen without uncertainty prediction activated.")
            else:
                logging.warning("Feature {} is unknown. Won't include it in ME gating.".format(feature))
        # Check if there is at least one input to the network
        assert(input_dim > 0)
        self.input_dim = input_dim
        # Output dimensions
        self.n_experts = n_experts
        super().__init__(n_experts, "ME weighting", model_path)

    def create_model(self):
        """Create a new MLP ME model based on the model structure defined in the config file."""
        self.mlp_model = me_mlp_model_factory(input_dim=self.input_dim, 
                                              n_experts=self.n_experts, 
                                              is_uncertainty_prediction=self.is_uncertainty_prediction, 
                                              direct_uncertainty_output=self.direct_uncertainty_output,
                                              **self.model_structure)
        
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
        self.load_calibration()
        logging.info(self.mlp_model.summary())

    def train_network(self, inputs, target, predictions, masks, inputs_eval, target_eval, predictions_eval, masks_eval, **kwargs):
        """Train the mixture of expert network.
        
        Create the network architecture.
        Transform the data to fit the network architecture.

        Args:
            inputs (np.array):          All MLP inputs, shape: [n_tracks, 2*n_mlp_inp_points]
            targets (np.array):         All target values of the given dataset, shape: [n_tracks, 2]
            predictions (np.array):     All predictions for all experts, shape: [n_experts, n_tracks, 2 or 4]
            masks (np.array):           Masks for each expert, shape: [n_experts, n_tracks]
        """
        ## Create Model
        self.create_model()
        # Define Learning rate reduction
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.base_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)
        # Define Adam
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
        # Define train step fuction
        train_step_fn = train_step_generator(model=self.mlp_model, 
                                            optimizer=optimizer,
                                            is_uncertainty_prediction = self.is_uncertainty_prediction,
                                            direct_uncertainty_output = self.direct_uncertainty_output,
                                            is_one_out_of_n_selector = self.is_one_out_of_n_selector,
                                            correlations = tf.convert_to_tensor(self.corr))
        # Create a mask for ME network
        mlp_mask = np.all(inputs>0, axis=1)
        mlp_mask_eval = np.all(inputs_eval>0, axis=1)
        assert(inputs.shape[0]==target.shape[0])
        assert(inputs_eval.shape[0]==target_eval.shape[0])
        assert(predictions.shape[0]==self.n_experts)
        # Add the expert uncertainty prediction as input if activated
        if self.use_uncertainty_prediction_as_input:
            inputs = np.concatenate((inputs, np.swapaxes(predictions, 0, 1)[:,:,2], np.swapaxes(predictions, 0, 1)[:,:,3]), axis=-1)
            inputs_eval = np.concatenate((inputs_eval, np.swapaxes(predictions_eval, 0, 1)[:,:,2], np.swapaxes(predictions_eval, 0, 1)[:,:,3]), axis=-1)
        n_inputs = inputs.shape[1]
        n_target = 2
        n_experts = self.n_experts
        # Combine Input, target, predictions and masks to one numpy array
        if not self.is_uncertainty_prediction:
            train_data = np.concatenate((inputs, 
                                        target[:,0:2],
                                        np.swapaxes(predictions, 0, 1)[:,:,0], 
                                        np.swapaxes(predictions, 0, 1)[:,:,1], 
                                        np.swapaxes(masks, 0, 1), 
                                        mlp_mask[:,np.newaxis]), axis=-1)
            eval_data = np.concatenate((inputs_eval, 
                                        target_eval[:,0:2],
                                        np.swapaxes(predictions_eval, 0, 1)[:,:,0], 
                                        np.swapaxes(predictions_eval, 0, 1)[:,:,1], 
                                        np.swapaxes(masks_eval, 0, 1), 
                                        mlp_mask_eval[:,np.newaxis]), axis=-1)
        else:
            train_data = np.concatenate((inputs, 
                                        target[:,0:2], 
                                        np.swapaxes(predictions, 0, 1)[:,:,0], 
                                        np.swapaxes(predictions, 0, 1)[:,:,1], 
                                        np.swapaxes(predictions, 0, 1)[:,:,2], 
                                        np.swapaxes(predictions, 0, 1)[:,:,3], 
                                        np.swapaxes(masks, 0, 1), 
                                        mlp_mask[:,np.newaxis]), 
                                        axis=-1)
            eval_data = np.concatenate((inputs_eval, 
                                        target_eval[:,0:2], 
                                        np.swapaxes(predictions_eval, 0, 1)[:,:,0], 
                                        np.swapaxes(predictions_eval, 0, 1)[:,:,1], 
                                        np.swapaxes(predictions_eval, 0, 1)[:,:,2], 
                                        np.swapaxes(predictions_eval, 0, 1)[:,:,3], 
                                        np.swapaxes(masks_eval, 0, 1), 
                                        mlp_mask_eval[:,np.newaxis]), 
                                        axis=-1)
        # Create TF Dataset
        raw_train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        raw_eval_dataset = tf.data.Dataset.from_tensor_slices(eval_data)
        minibatches_train = raw_train_dataset.batch(self.batch_size, drop_remainder=True)
        minibatches_eval = raw_eval_dataset.batch(self.batch_size, drop_remainder=True)
        def split_input_target_separation(chunk):
            # split the tensor
            input_seq = chunk[:, :n_inputs]
            target_seq = chunk[:, n_inputs:n_inputs+n_target]
            n_prediction_cols = 4*n_experts if self.is_uncertainty_prediction else 2*n_experts
            spatial_predictions = chunk[:, n_inputs+n_target:n_inputs+n_target+n_experts]
            temporal_predictions = chunk[:, n_inputs+n_target+n_experts:n_inputs+n_target+2*n_experts]
            if self.is_uncertainty_prediction:
                spatial_uncertainty_prediction = chunk[:,n_inputs+n_target+2*n_experts:n_inputs+n_target+3*n_experts]
                temporal_uncertainty_prediction = chunk[:,n_inputs+n_target+3*n_experts:n_inputs+n_target+4*n_experts]
            expert_mask = chunk[:, n_inputs+n_target+n_prediction_cols:-1]
            mlp_mask = chunk[:, -1]
            if not self.is_uncertainty_prediction:
                return input_seq, target_seq, spatial_predictions, temporal_predictions, expert_mask, mlp_mask
            else:
                return input_seq, target_seq, spatial_predictions, temporal_predictions, spatial_uncertainty_prediction, temporal_uncertainty_prediction, expert_mask, mlp_mask
        dataset_train = minibatches_train.map(split_input_target_separation)
        dataset_eval = minibatches_eval.map(split_input_target_separation)
        
        ## Values for training logging in tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '_train_me_gating' + '/train'
        eval_log_dir = 'logs/gradient_tape/' + current_time + '_train_me_gating' + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)

        ## Training
        for epoch in range(self.n_epochs):
            # Iterate over the batches of the train dataset.
            train_iter = iter(dataset_train)
            if not self.is_uncertainty_prediction:
                (train_inputs, train_target, train_spatial_predictions, train_temporal_predictions, train_masks, mlp_mask) = train_iter.next()
                train_expert_predictions = tf.concat([tf.expand_dims(train_spatial_predictions, axis = -1), tf.expand_dims(train_temporal_predictions, axis = -1)], axis = -1)
            else:
                (train_inputs, train_target, train_spatial_predictions, train_temporal_predictions, train_spatial_uncertainty_prediction, train_temporal_uncertainty_prediction, train_masks, mlp_mask) = train_iter.next()
                train_expert_predictions = tf.concat([tf.expand_dims(train_spatial_predictions, axis = -1), 
                                                      tf.expand_dims(train_temporal_predictions, axis = -1),
                                                      tf.expand_dims(train_spatial_uncertainty_prediction, axis = -1),
                                                      tf.expand_dims(train_temporal_uncertainty_prediction, axis = -1)], axis = -1)
            while True:
                # Predict weights for experts 
                weights, loss = train_step_fn(train_inputs, train_target, train_expert_predictions, train_masks, mlp_mask)
                train_loss(loss)
                # Load next batch
                try:
                    if not self.is_uncertainty_prediction:
                        (train_inputs, train_target, train_spatial_predictions, train_temporal_predictions, train_masks, mlp_mask) = train_iter.next()
                        train_expert_predictions = tf.concat([tf.expand_dims(train_spatial_predictions, axis = -1), tf.expand_dims(train_temporal_predictions, axis = -1)], axis = -1)
                    else:
                        (train_inputs, train_target, train_spatial_predictions, train_temporal_predictions, train_spatial_uncertainty_prediction, train_temporal_uncertainty_prediction, train_masks, mlp_mask) = train_iter.next()
                        train_expert_predictions = tf.concat([tf.expand_dims(train_spatial_predictions, axis = -1), 
                                                            tf.expand_dims(train_temporal_predictions, axis = -1),
                                                            tf.expand_dims(train_spatial_uncertainty_prediction, axis = -1),
                                                            tf.expand_dims(train_temporal_uncertainty_prediction, axis = -1)], axis = -1)
                except StopIteration:
                    break
            

            # Write training results
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
            # Console output
            template = 'Epoch {}, Train Loss: {}'
            logging.info(template.format(epoch+1, train_loss.result().numpy()))
            # Reset metrics every epoch
            train_loss.reset_states(); 

            # Run trained models on the test set every n epochs
            if (epoch + 1) % self.evaluate_every_n_epochs == 0 \
                    or (epoch + 1) == self.n_epochs:
                # Iterate over the batches of the eval dataset.
                eval_iter = iter(dataset_eval)
                if not self.is_uncertainty_prediction:
                    (eval_inputs, eval_target, eval_spatial_predictions, eval_temporal_predictions, eval_masks, eval_mlp_mask) = eval_iter.next()
                    eval_expert_predictions = tf.concat([tf.expand_dims(eval_spatial_predictions, axis = -1), tf.expand_dims(eval_temporal_predictions, axis = -1)], axis = -1)
                else:
                    (eval_inputs, eval_target, eval_spatial_predictions, eval_temporal_predictions, eval_spatial_uncertainty_prediction, eval_temporal_uncertainty_prediction, eval_masks, eval_mlp_mask) = eval_iter.next()
                    eval_expert_predictions = tf.concat([tf.expand_dims(eval_spatial_predictions, axis = -1), 
                                                        tf.expand_dims(eval_temporal_predictions, axis = -1),
                                                        tf.expand_dims(eval_spatial_uncertainty_prediction, axis = -1),
                                                        tf.expand_dims(eval_temporal_uncertainty_prediction, axis = -1)], axis = -1)
            
                while True:
                    weights, loss = train_step_fn(eval_inputs, eval_target, eval_expert_predictions, eval_masks, eval_mlp_mask, training=False)
                    eval_loss(loss)
                    try:
                        if not self.is_uncertainty_prediction:
                            (eval_inputs, eval_target, eval_spatial_predictions, eval_temporal_predictions, eval_masks, mlp_mask) = eval_iter.next()
                            eval_expert_predictions = tf.concat([tf.expand_dims(eval_spatial_predictions, axis = -1), tf.expand_dims(eval_temporal_predictions, axis = -1)], axis = -1)
                        else:
                            (eval_inputs, eval_target, eval_spatial_predictions, eval_temporal_predictions, eval_spatial_uncertainty_prediction, eval_temporal_uncertainty_prediction, eval_masks, mlp_mask) = eval_iter.next()
                            eval_expert_predictions = tf.concat([tf.expand_dims(eval_spatial_predictions, axis = -1), 
                                                                tf.expand_dims(eval_temporal_predictions, axis = -1),
                                                                tf.expand_dims(eval_spatial_uncertainty_prediction, axis = -1),
                                                                tf.expand_dims(eval_temporal_uncertainty_prediction, axis = -1)], axis = -1)
                    except StopIteration:
                        break
                # Write evaluation results
                with eval_summary_writer.as_default():
                    tf.summary.scalar('loss', eval_loss.result(), step=epoch)
                # Console output
                template = 'Epoch {}, eval Loss: {}'
                logging.info(template.format(epoch+1, eval_loss.result().numpy()))
                # Reset metrics every epoch
                eval_loss.reset_states()

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

    def get_weights(self, batch_size):
        """Return a weight vector.
        
        The weights sum to 1.
        All Weights are > 0.

        Returns:
            np.array with weights of shape [n_experts, batch_size]
        """
        pass

    def get_masked_weights(self, masks, inputs):
        """Return an equal weights vector for all non masked experts.
        
        The weights sum to 1.
        All Weights are > 0 if the expert is non masked at an instance.

        If the mask value at an instance is 0, the experts weight is 0.
        
        Args:
            masks (np.array):  Mask array with shape [n_experts, n_tracks]
            inputs (np.array): Data input in MLP separation format, shape: [n_tracks, 2*n_mlp_inp]
      
        Returns:
            np.array with weights of shape [mask.shape, 2] -> 2 standing for the two dimensions spatial and temporal
        """
        n_experts = masks.shape[0]
        weights = self.mlp_model(inputs, training = False)
        weights = np.array(weights)
        # Handle the cases where there are less than n measurements
        ignore = np.all(inputs==0, axis=1)
        if sum(ignore) > 0:
            # Find best expert for non ignore cases
            best_expert_spatial = np.argmax(np.mean(weights[0, ~ignore], axis=0))
            best_expert_temporal = np.argmax(np.mean(weights[1, ~ignore], axis=0))
            # If the best expert is masked out where there are less than n measurements, we need to use SE weighting
            if np.any(masks[best_expert_spatial, ignore] == 0):
                default_spatial = 1/n_experts * np.ones(n_experts)
            else:
                # Otherwise we can use the best expert
                default_spatial = np.zeros(n_experts)
                default_spatial[best_expert_spatial] = 1
            # Temporal
            if np.any(masks[best_expert_temporal, ignore] == 0):
                default_temporal = 1/n_experts * np.ones(n_experts)
            else:
                default_temporal = np.zeros(n_experts)
                default_temporal[best_expert_temporal] = 1
            # Replace incorrect positions
            weights[0, ignore, :] = default_spatial
            weights[1, ignore, :] = default_temporal
        masked_weights = mask_weights(weights, np.swapaxes(masks,0,1))
        masked_weights = masked_weights.numpy()
        return np.swapaxes(masked_weights, 0, 1)

    def get_masked_weights_and_uncertainty(self, masks, log_variance_predictions, inputs=None):
        """Return an equal weights vector for all non masked experts.
        
        The weights sum to 1.
        All Weights are > 0 if the expert is non masked at an instance.

        If the mask value at an instance is 0, the experts weight is 0.
        
        Args:
            masks (np.array):  Mask array with shape [n_experts, n_tracks]
            inputs (np.array): Data input in MLP separation format, shape: [n_tracks, 2*n_mlp_inp]
            log_variance_predictions (np.array): All uncertainty predictions for all experts, shape: [n_experts, n_tracks, 2]

        Returns:
            weights: np.array with weights of shape [mask.shape, 2] -> 2 standing for the two dimensions spatial and temporal
            uncertainty: np.array containing log(variance) with shape [n_tracks, 2]
        """
        n_experts = masks.shape[0]
        if self.use_uncertainty_prediction_as_input:
            inputs = np.concatenate((inputs, np.swapaxes(log_variance_predictions, 0, 1)[:,:,0], np.swapaxes(log_variance_predictions, 0, 1)[:,:,1]), axis=-1) 
        outputs = self.mlp_model(inputs, training = False)
        weights = np.array(outputs[:2])
        
        
        masked_weights = mask_weights(weights, np.swapaxes(masks,0,1))
        masked_weights = masked_weights.numpy()
        masked_weights = np.swapaxes(masked_weights, 0, 1)
        if not self.direct_uncertainty_output:
            # Combine variance prediction
            variance_predictions = np.exp(log_variance_predictions)
            # var[k,l] = sum_{i} sum_{j} (w[i,k,l]*cov[l,i,j]*w[j,k,l])
            #       - sum_{i} (w[i,k,l]**2 * cov[l,i,i])
            #       + sum_{j} (w[i,k,l]**2 * var_pred[i,k,l])
            combined_var_einsum = np.einsum('ikl,jkl,lij,ikl,jkl->kl', masked_weights, masked_weights, self.corr, np.sqrt(variance_predictions), np.sqrt(variance_predictions))
            combined_log_var = np.log(combined_var_einsum) 
        else:
            combined_log_var = np.swapaxes(np.array(outputs[2:]), 0,1)[:,:,0]
        # Handle the cases where there are less than n measurements
        ignore = np.all(inputs==0, axis=1)
        if sum(ignore) > 0:
            # Find best expert for non ignore cases
            best_expert_spatial = np.argmax(np.mean(weights[0, ~ignore], axis=0))
            best_expert_temporal = np.argmax(np.mean(weights[1, ~ignore], axis=0))
            # If the best expert is masked out where there are less than n measurements, we need to use SE weighting
            if np.any(masks[best_expert_spatial, ignore] == 0):
                default_spatial = 1/n_experts * np.ones(n_experts)
            else:
                # Otherwise we can use the best expert
                default_spatial = np.zeros(n_experts)
                default_spatial[best_expert_spatial] = 1
            # Temporal
            if np.any(masks[best_expert_temporal, ignore] == 0):
                default_temporal = 1/n_experts * np.ones(n_experts)
            else:
                default_temporal = np.zeros(n_experts)
                default_temporal[best_expert_temporal] = 1
            # Replace incorrect positions
            weights[0, ignore, :] = default_spatial
            weights[1, ignore, :] = default_temporal
            combined_log_var[ignore, 0] = log_variance_predictions[best_expert_spatial, ignore, 0]
            combined_log_var[ignore, 1] = log_variance_predictions[best_expert_temporal, ignore, 1]
        return masked_weights, combined_log_var


"""Model creation and training functionality"""

def me_mlp_model_factory(input_dim, n_experts, is_uncertainty_prediction = False, direct_uncertainty_output = False,
                         prediciton_input_dim=2, features=["pos"], layers=[16, 16, 16], activation='leakly_relu',
                         l1_regularization=0, l2_regularization=0):
    """Create a new keras MLP model

    Args:
        input_dim (int):            The number of inputs to the model
        n_experts (int):            The number of experts
        is_uncertainty_prediction (Boolean): Predict uncertainty of predictions. 
        direct_uncertainty_output (Boolean): If this is set to False, the uncertainty is calculated with the weighted variance predictions
                                             If this is set to True, the uncertainty is directly outputted. (Increases the output dimension by 2.)
        prediciton_input_dim (int): Should be 2 for spatial and temporal position.
        features (list):            List of String. Features for MLP. Possibilities:
                                        "pos":  x and y position of current measurement.
                                        "id":   Current track id - Number of measurements in track
                                        "uncertainty_prediction": The uncertainty prediction of each expert
        layers (list):              List of int. Number of nodes and layers
        activation (String):        Activation function for layers
        l1_regularization (double): L1 Regularization factor
        l2_regularization (double): L2 Regularization factor
    
    Returns: 
        the model
    """
    # Define regularization
    regularization = tf.keras.regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)
    # The input layer
    inputs = tf.keras.Input(shape=(input_dim,), name='me_input')

    # Add hidden layers
    is_first = True
    is_doubled = False
    c=0
    for n_Neurons in layers:
        # Add layer
        if is_first:
            input_layer = inputs
            is_first=False
        else:
            input_layer = x
        if isinstance(n_Neurons, list):
            # We allow a split of the network in two for the spatial and temporal branch
            assert(len(n_Neurons)==2, "Only a split in 2 branches is allowed.")
            input_layer = x_spatial if is_doubled else input_layer
            x_spatial = tf.keras.layers.Dense(n_Neurons[0], kernel_initializer='he_normal', name='dense_spatial_{}'.format(c),
                                              kernel_regularizer=regularization, bias_regularizer=regularization)(input_layer)
            input_layer = x_temporal if is_doubled else input_layer
            x_temporal = tf.keras.layers.Dense(n_Neurons[1], kernel_initializer='he_normal', name='dense_temporal_{}'.format(c),
                                              kernel_regularizer=regularization, bias_regularizer=regularization)(input_layer)
            # Add activation function to layer
            if activation == 'leakly_relu':
                x_spatial = tf.keras.layers.LeakyReLU(alpha=0.01)(x_spatial)
                x_temporal = tf.keras.layers.LeakyReLU(alpha=0.01)(x_temporal)
            else:
                logging.warning("Activation function {} not implemented yet :(".format(activation))
            is_doubled=True
        else:
            assert(is_doubled==False, "Once you split the branches, you shall not go back. (It would be possible but I was too lazy to implement it.)")
            x = tf.keras.layers.Dense(n_Neurons, kernel_initializer='he_normal', name='dense_{}'.format(c),
                                      kernel_regularizer=regularization, bias_regularizer=regularization)(input_layer)
            # Add activation function to layer
            if activation == 'leakly_relu':
                x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
            else:
                logging.warning("Activation function {} not implemented yet :(".format(activation))
        c+=1
    # Output layers = 2 Dense layers with seperate softmax activation functions
    input_layer = x_spatial if is_doubled else x
    spatial_weights = tf.keras.layers.Dense(n_experts, name='spatial_weights',
                                            kernel_regularizer=regularization, bias_regularizer=regularization)(input_layer)
    spatial_weights = tf.keras.layers.Softmax()(spatial_weights)
    # If direct_uncertainty_output is activated, we need an additional output per dimension for the uncertainty
    if direct_uncertainty_output:
        spatial_uncertainty = tf.keras.layers.Dense(1, name='spatial_uncertainty',
                                                    kernel_regularizer=regularization, bias_regularizer=regularization)(input_layer)
    input_layer = x_temporal if is_doubled else x
    temporal_weights = tf.keras.layers.Dense(n_experts, name='temporal_weights',
                                             kernel_regularizer=regularization, bias_regularizer=regularization)(input_layer)
    temporal_weights = tf.keras.layers.Softmax()(temporal_weights)
    if direct_uncertainty_output:
        temporal_uncertainty = tf.keras.layers.Dense(1, name='temporal_uncertainty',
                                                     kernel_regularizer=regularization, bias_regularizer=regularization)(input_layer)
        model = tf.keras.Model(inputs=inputs, outputs=[spatial_weights, temporal_weights, spatial_uncertainty, temporal_uncertainty])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=[spatial_weights, temporal_weights])
    return model

def train_step_generator(model, optimizer, is_uncertainty_prediction = False, direct_uncertainty_output=False,
                         is_one_out_of_n_selector = False, correlations = []):
    """Build a function which returns a computational graph for tensorflow.

    This function can be called to train the given model with the given optimizer.

    Args:
        model:      model according to estimator api
        optimizer:  tf estimator
        is_uncertainty_prediction (Boolean): Predict uncertainty of predictions. 
        direct_uncertainty_output (Boolean): If this is set to False, the uncertainty is calculated with the weighted variance predictions
                                             If this is set to True, the uncertainty is directly outputted. (Increases the output dimension by 2.)
        is_one_out_of_n_selector (Boolean): Only pick the highest ranked expert for the prediction
        correlations (tf.Tensor): The expert prediction error correlations, 
                                trained in gating_network.train_correlations(), converted to tf.Tensor,
                                Shape: [n_dim (2), n_experts, n_experts]

    Returns:
        function which can be called to train the given model with the given optimizer
    """
    @tf.function
    def train_step(inp, target, expert_predictions, mask, mlp_mask, training=True):
        with tf.GradientTape() as tape:
            #target = K.cast(target, tf.float64)
            # get [n_experts spatial weights, n_experts temporal weights] from MLP
            outputs = model(inp, training=training)
            weights = outputs[:2]

            # Mask the weights
            masked_weights = mask_weights(weights, mask)
            if is_one_out_of_n_selector:
                # Highest ranked expert
                best_expert_ids = tf.math.argmax(masked_weights, axis=1)
                # Best expert gets weight = 1 and all the rest gets weight = 0
                masked_weights = tf.transpose(tf.one_hot(best_expert_ids, expert_predictions.get_shape()[1], dtype=tf.float64), [0, 2, 1])
            if not is_uncertainty_prediction:
                loss = weighted_sum_mse_loss(masked_weights, expert_predictions, target, mlp_mask)
            else:
                if not direct_uncertainty_output:
                    loss = weighted_sum_nll_loss(masked_weights, expert_predictions, target, mlp_mask, correlations)
                else:
                    uncertainty_prediction = tf.transpose(tf.convert_to_tensor(outputs[2:]), [1,0,2])[:,:,0]
                    loss = nll_loss(masked_weights, uncertainty_prediction, expert_predictions, target, mlp_mask)
        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return weights, loss

    return train_step

def mask_weights(weights, mask):
    """Mask the tf weights vector.

    Args:
        weights (list):   The outputs of the MLP network, shape: [[batch_size, n_experts], [batch_size, n_experts]]
        mask (tf.Tensor): The mask for all experts, shape:       [batch_size, n_experts]

    Returns:
        masked_weights (tf.Tensor):  Weights for spatial and temporal prediction, shape: [batch_size, n_experts, 2]
    """
    masked_weights_spatial = tf.multiply(weights[0], mask)
    masked_weights_spatial, _ = tf.linalg.normalize(masked_weights_spatial, ord=1, axis=1)
    masked_weights_temporal = tf.multiply(weights[1], mask)
    masked_weights_temporal, _ = tf.linalg.normalize(masked_weights_temporal, ord=1, axis=1)
    masked_weights = tf.concat([tf.expand_dims(masked_weights_spatial, axis = -1), tf.expand_dims(masked_weights_temporal, axis = -1)], axis = -1)
    return masked_weights

def weighted_sum_mse_loss(weights, expert_predictions, target, mask):
    """Return MSE for weighted expert predictions.

    Args:
        expert_predictions (tf.Tensor): The expert predictions, shape: [batch_size, n_experts, 2]
        weights (tf.Tensor):            The outputs of the MLP network, shape: [batch_size, n_experts, 2]
        target (tf.Tensor):             The target, shape: [batch_size, 2]
        mask (tf.Tensor):               Mask entries that are valid, shape: [batch_size,]
    Returns:
        loss (float64)
    """
    weighted_prediction = tf.einsum('ijk,ijk->ik', expert_predictions, weights)
    double_mask = tf.repeat(tf.expand_dims(mask, axis = -1), 2, axis = -1)
    return tf.reduce_sum(tf.pow(target-weighted_prediction,2)*double_mask)/tf.reduce_sum(double_mask)

def weighted_sum_nll_loss(weights, expert_predictions, target, mask, correlations):
    """Return negative log likelihood loss for weighted expert predictions.

    Args:
        expert_predictions (tf.Tensor): The expert predictions, shape: [batch_size, n_experts, 4]
        weights (tf.Tensor):            The outputs of the MLP network, shape: [batch_size, n_experts, 2]
        target (tf.Tensor):             The target, shape: [batch_size, 2]
        mask (tf.Tensor):               Mask entries that are valid, shape: [batch_size,]
        correlations (tf.Tensor):       The expert prediction error correlations, 
                                        trained in gating_network.train_correlations(), converted to tf.Tensor,
                                        Shape: [n_dim (2), n_experts, n_experts]
    Returns:
        loss (float64)
    """
    # Mean prediction 
    mean_predictions = expert_predictions[:,:,:2]
    # Variance prediction
    variance_predictions = tf.exp(expert_predictions[:,:,2:])
    std_predictions = tf.sqrt(variance_predictions)
    # Create prediction of means
    weighted_prediction = tf.einsum('ijk,ijk->ik', mean_predictions, weights)
    double_mask = tf.repeat(tf.expand_dims(mask, axis = -1), 2, axis = -1)
    # Create prediction of uncertainties
    # var[k,l] = sum_{i} sum_{j} (w[k,i,l]*cov[l,i,j]*w[k,j,l])
    #       - sum_{i} (w[k,i,l]**2 * cov[l,i,i])
    #       + sum_{j} (w[k,i,l]**2 * var_pred[k,i,l])
    combined_var = tf.einsum('kil,kjl,lij,kil,kjl->kl', weights, weights, correlations, std_predictions, std_predictions)
    log_var = tf.math.log(combined_var)
    L = tf.pow(target-weighted_prediction,2) / combined_var + log_var
    loss = tf.reduce_sum(L*double_mask) / tf.reduce_sum(double_mask)
    return loss

def nll_loss(weights, uncertainty_prediction, expert_predictions, target, mask):
    """Return negative log likelihood loss for weighted expert mean predictions and direct uncertainty prediction.

    Args:
        expert_predictions (tf.Tensor):     The expert predictions, shape: [batch_size, n_experts, 4]
        weights (tf.Tensor):                The outputs of the MLP network, shape: [batch_size, n_experts, 2]
        uncertainty_prediction (tf.Tensor): The predicted log(variance) output of the gating network, shape: [batch_size, 1, 2]
        target (tf.Tensor):                 The target, shape: [batch_size, 2]
        mask (tf.Tensor):                   Mask entries that are valid, shape: [batch_size,]

    Returns:
        loss (float64)
    """
    mean_predictions = expert_predictions[:,:,:2]
    weighted_prediction = tf.einsum('ijk,ijk->ik', mean_predictions, weights)
    double_mask = tf.repeat(tf.expand_dims(mask, axis = -1), 2, axis = -1)
    L = tf.pow(target-weighted_prediction,2) / tf.math.exp(uncertainty_prediction) + uncertainty_prediction
    loss = tf.reduce_sum(L*double_mask) / tf.reduce_sum(double_mask)
    return loss
