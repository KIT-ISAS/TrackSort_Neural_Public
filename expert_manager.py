"""Expert Manager.

Todo:
    * Add train and test method
    * Convert np representation to tensor representation for mixture of experts
    * Understand state buffering for RNN
"""

import logging
import tensorflow as tf
import numpy as np
import code
import time

from tensorflow.keras import backend as K

#tf.keras.backend.set_floatx('float64')

from rnn_model import RNN_Model
from mlp_model import MLP_Model
from kf_model import KF_Model, KF_State
from cv_model import CV_Model, CV_State
from ca_model import CA_Model, CA_State
from expert import Expert, Expert_Type

class Expert_Manager(object):
    """The expert manager handles all forecasting experts.

    Handles the experts and their states in continuous forecasting.
    Provides train and test methods.

    Attributes:
        current_states (list):  Stores the internal states of all tracks for each model.
                                    State structure may vary for each type of expert.
                                    current_states[batch_nr] = [Batch States[...]]
        n_experts (int):        The number of experts in the expert bank.
    """
    
    def __init__(self, expert_config, is_loaded, model_path="", batch_size=64, num_time_steps=0, n_mlp_features=10, n_mlp_features_separation_prediction = 7, x_pred_to = 1550, time_normalization = 22.):
        """Initialize an expert manager.

        Creates the expert models.
        Initializes attributes.

        Args:
            expert_config (dict): The configuration dictionary of all experts
            is_loaded (Boolean):  True for loading models, False for creating new models
            model_path (String):  The path of the models if is_loaded is True
            batch_size (int):     The batch size of the data
            num_time_steps (int): The number of timesteps in the longest track
            n_mlp_features (int): The numper of features for MLP tracking
            n_mlp_features_separation_prediction (int): The number of features for MLP separation prediction networks
            x_pred_to (double):   The x position of the nozzle array (only needed for kf separation prediction)
            time_normalization (double): Time normalization constant (only needed for kf separation prediction)
        """
        self.expert_config = expert_config
        self.batch_size = batch_size
        # List of list of states -> Each model has its own list of current states (= particles)
        self.current_states = []
        self.experts = []
        self.separation_experts = []
        self.create_models(is_loaded, model_path, batch_size, num_time_steps, n_mlp_features, n_mlp_features_separation_prediction, x_pred_to, time_normalization)
        self.n_experts = len(self.experts)

    def create_models(self, is_loaded, model_path="", batch_size=64, num_time_steps=0, n_mlp_features = 5, n_mlp_features_separation_prediction = 7, x_pred_to = 1550, time_normalization = 22.):
        """Create list of experts.

        Creat experts based on self.expert_cofig.
        Add empty list to list of states for each expert.
        Load experts from model_path if is_loaded is True.
        Create new experts if is_loaded is False.

        We multiply the number of MLP features by 2!
            --> 1 point equals 2 coordinates (x and y)

        Args:
            is_loaded (Boolean):            True for loading models, False for creating new models
            model_path (String):            The path of the models if is_loaded is True
            batch_size (int):               The batch size of the data
            num_time_steps (int):           The number of timesteps in the longest track
            n_mlp_features (int):           The number of features for MLP networks
            n_mlp_features_separation_prediction (int): The number of features for MLP separation prediction networks
            x_pred_to (double):             The x position of the nozzle array (only needed for kf separation prediction)
            time_normalization (double):    Time normalization constant (only needed for kf separation prediction)
        """
        for expert_name in self.expert_config:
            expert = self.expert_config.get(expert_name)
            expert_type = expert.get("type")
            if expert_type == 'RNN':
                model_path = expert.get("model_path")
                if "is_separator" in expert and expert.get("is_separator"):
                    # Create RNN model for separation prediction
                    rnn_model = RNN_Model(False, expert_name, model_path, expert.get("options"))
                    if is_loaded:
                        rnn_model.load_model()
                    else:
                        only_last_timestep_additional_loss = False if "only_last_timestep_additional_loss" not in expert else expert.get("only_last_timestep_additional_loss")
                        rnn_model.create_model(batch_size=batch_size, 
                                               num_time_steps=num_time_steps,
                                               only_last_timestep_additional_loss=only_last_timestep_additional_loss)
                    self.separation_experts.append(rnn_model)
                else:
                    # Create RNN model
                    rnn_model = RNN_Model(True, expert_name, model_path, expert.get("options"))
                    if is_loaded:
                        rnn_model.load_model()
                    else:
                        rnn_model.create_model(batch_size, num_time_steps)
                    self.experts.append(rnn_model)
                    self.current_states.append([])
            elif expert_type == 'KF':
                # Create Kalman filter model
                sub_type = expert.get("sub_type")
                if sub_type == 'CV':
                    # Create constant velocity model
                    kf_model = CV_Model(expert_name, x_pred_to, time_normalization, **expert.get("model_options"), default_state_options=expert.get("state_options"))
                elif sub_type == 'CA':
                    # Create constant acceleration model
                    kf_model = CA_Model(expert_name, x_pred_to, time_normalization, **expert.get("model_options"), default_state_options=expert.get("state_options"))
                else:
                    logging.warning("Kalman filter subtype " + sub_type + " not supported. Will not create model.") 
                    continue
                if "is_separator" in expert and expert.get("is_separator"):
                    self.separation_experts.append(kf_model)
                else:
                    self.experts.append(kf_model)
                    self.current_states.append([])
            elif expert_type=='MLP':
                model_path = expert.get("model_path")
                
                if "is_separator" in expert and expert.get("is_separator"):
                    # Create MLP model for separation prediction
                    mlp_model = MLP_Model(expert_name, model_path, False, expert.get("options"))
                    if is_loaded:
                        mlp_model.load_model()
                    else:
                        mlp_model.create_model(2 * n_mlp_features_separation_prediction)
                    self.separation_experts.append(mlp_model)
                else:
                    # Create MLP model for tracking
                    mlp_model = MLP_Model(expert_name, model_path, True, expert.get("options"))
                    if is_loaded:
                        mlp_model.load_model()
                    else:
                        mlp_model.create_model(2 * n_mlp_features)
                    self.experts.append(mlp_model)
                    self.current_states.append([])
                
            else:
                logging.warning("Expert type " + expert_type + " not supported. Will not create model.")

    def save_models(self):
        """Save all models to their model paths."""
        for expert in self.experts:
            expert.save_model()

    def train_batch(self, mlp_conversion_func,
                    seq2seq_inp = None, seq2seq_target = None, 
                    mlp_inp = None, mlp_target = None):
        """Train one batch for all experts in tracking.

        The training information of each model should be provided in the expert configuration.
        Kalman filters and RNNs need a different data format than MLPs.

        Args:
            mlp_conversion_func:   Function to convert MLP format to track format
            **_inp (tf.Tensor):    Input tensor of tracks
            **_target (tf.Tensor): Target tensor of tracks

        Returns:
            predictions (list): Predictions for each expert
        """
        prediction_list = []
        for expert in self.experts:
            if expert.type == Expert_Type.KF or expert.type == Expert_Type.RNN:
                prediction = expert.train_batch(seq2seq_inp, seq2seq_target) 
                if tf.is_tensor(prediction):
                    prediction = prediction.numpy()
            elif expert.type == Expert_Type.MLP:
                prediction = expert.train_batch(mlp_inp, mlp_target)
                if tf.is_tensor(prediction):
                    prediction = prediction.numpy()
                prediction = mlp_conversion_func(prediction)
            prediction_list.append(prediction)
            
        return prediction_list

    def train_batch_separation_prediction(self,                     
                    seq2seq_inp = None, seq2seq_target = None, 
                    seq2seq_tracking_mask = None, seq2seq_separation_mask = None,
                    mlp_inp = None, mlp_target = None, mlp_mask=None):
        """Train one batch for all experts in separation prediction.

        TODO: Consider merging this funftion with train_batch

        The training information of each model should be provided in the expert configuration.
        Kalman filters and RNNs need a different data format than MLPs.

        Args:
            **_inp (tf.Tensor):    Input tensor of tracks
            **_target (tf.Tensor): Target tensor of tracks
            seq2seq_tracking_mask (tf.Tensor): Mask the valid time steps for tracking
            seq2seq_separation_mask (tf.Tensor): Mask the valid time step(s) for the separation prediction
            mlp_mask (tf.Tensor):   Mask the tracks that have less than n points

        Returns:
            predictions (list): Predictions for each expert
            spatial_losses (list): Spacial loss for each expert 
            temporal_losses (list): Temporal loss for each expert
            spatial_maes (list): Spacial mae for each expert
            temporal_maes (list): Temporal mae for each expert
        """
        prediction_list = []
        spatial_losses = []
        temporal_losses = []
        spatial_maes = []
        temporal_maes = []
        for expert in self.separation_experts:
            if expert.type == Expert_Type.KF or expert.type == Expert_Type.RNN:
                prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae = expert.train_batch_separation_prediction(seq2seq_inp, seq2seq_target, seq2seq_tracking_mask, seq2seq_separation_mask) 
            elif expert.type == Expert_Type.MLP:
                prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae = expert.train_batch_separation_prediction(mlp_inp, mlp_target, mlp_mask)
            prediction_list.append(prediction)
            spatial_losses.append(spatial_loss)
            temporal_losses.append(temporal_loss)
            spatial_maes.append(spatial_mae)
            temporal_maes.append(temporal_mae)
            
        return prediction_list, spatial_losses, temporal_losses, spatial_maes, temporal_maes

    def test_batch_separation_prediction(self,                     
                    seq2seq_inp = None, seq2seq_target = None, 
                    seq2seq_tracking_mask = None, seq2seq_separation_mask = None,
                    mlp_inp = None, mlp_target = None, mlp_mask=None):
        """Test one batch for all experts in separation prediction.

        TODO: Consider merging this funftion with train_batch

        The training information of each model should be provided in the expert configuration.
        Kalman filters and RNNs need a different data format than MLPs.

        Args:
            **_inp (tf.Tensor):    Input tensor of tracks
            **_target (tf.Tensor): Target tensor of tracks
            seq2seq_tracking_mask (tf.Tensor): Mask the valid time steps for tracking
            seq2seq_separation_mask (tf.Tensor): Mask the valid time step(s) for the separation prediction
            mlp_mask (tf.Tensor):   Mask the tracks that have less than n points

        Returns:
            predictions (list): Predictions for each expert
            spatial_losses (list): Spacial loss for each expert 
            temporal_losses (list): Temporal loss for each expert
            spatial_maes (list): Spacial mae for each expert
            temporal_maes (list): Temporal mae for each expert
        """
        prediction_list = []
        spatial_losses = []
        temporal_losses = []
        spatial_maes = []
        temporal_maes = []
        for expert in self.separation_experts:
            if expert.type == Expert_Type.KF or expert.type == Expert_Type.RNN:
                prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae = expert.test_batch_separation_prediction(seq2seq_inp, seq2seq_target, seq2seq_tracking_mask, seq2seq_separation_mask) 
            elif expert.type == Expert_Type.MLP:
                prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae = expert.test_batch_separation_prediction(mlp_inp, mlp_target, mlp_mask)
            prediction_list.append(prediction)
            spatial_losses.append(spatial_loss)
            temporal_losses.append(temporal_loss)
            spatial_maes.append(spatial_mae)
            temporal_maes.append(temporal_mae)
            
        return prediction_list, spatial_losses, temporal_losses, spatial_maes, temporal_maes

    def get_masks(self, mlp_conversion_func, k_mask_value, seq2seq_target, mlp_target):
        """Return masks for each expert.

        Args:
            mlp_conversion_func: Function to convert MLP format to track format
            k_mask_value:        Keras mask value to compare the target against to create the mask
            seq2seq_target:      Target for seq2seq data
            mlp_target:          Target for MLP data

        Returns:
            List of masks. One multidemensional mask for each expert.
        """
        masks = []
        for expert in self.experts:
            if expert.type == Expert_Type.KF or expert.type == Expert_Type.RNN:
                mask = K.all(K.equal(seq2seq_target, k_mask_value), axis=-1)
            elif expert.type == Expert_Type.MLP:
                track_target = mlp_conversion_func(mlp_target)
                mask = K.all(K.equal(track_target, k_mask_value), axis=-1)

            mask = 1 - K.cast(mask, tf.float64)
            #mask = K.cast(mask, tf.float64)
            masks.append(mask)
            
        return masks

    def get_masks_separation_prediction(self, mask_value, seq2seq_target, mlp_target):
        """Return masks for each expert.

        Args:
            mask_value:          mask value to compare the target against to create the mask
            seq2seq_target:      Target for seq2seq data
            mlp_target:          Target for MLP data

        Returns:
            List of masks. One multidemensional mask for each expert.
        """
        
        masks = []
        for expert in self.separation_experts:
            if expert.type == Expert_Type.KF or expert.type == Expert_Type.RNN:
                tracking_mask = 1 - tf.cast(tf.reduce_all(tf.equal(seq2seq_target, mask_value), axis=-1), tf.int32)
                mask_length = tf.reduce_sum(tracking_mask, axis=1) - 1
                mask = tf.one_hot(indices=mask_length, depth=seq2seq_target.shape.as_list()[1], off_value=0)
            elif expert.type == Expert_Type.MLP:
                mask = 1 - tf.cast(tf.reduce_all(tf.equal(mlp_target, mask_value), axis=-1), tf.int32)
            masks.append(tf.cast(mask, tf.float64))
            
        return masks

    def change_learning_rate(self, lr_change=1):
        """Change the learning rate of the certain models.

        Only change the learning rate of RNN models (right now).
        This can be used to lower the learning rate after n time steps to increase the accuracy.
        The change is implemented multiplicative. Set lr_change > 1 to increase and < 1 to decrease the lr.

        Args:
            lr_change (double): Change in learning rate (factorial)
        """
        for expert in self.experts:
            expert.change_learning_rate(lr_change)

    def test_batch(self, mlp_conversion_func, seq2seq_inp = None, mlp_inp = None):
        """Run predictions for all experts on a batch of test data.

        Args:
            inp (tf.Tensor): Input tensor of tracks

        Returns:
            prediction_list: Predictions of all experts
        """
        prediction_list = []
        for expert in self.experts:
            if expert.type == Expert_Type.KF or expert.type == Expert_Type.RNN:
                prediction = expert.predict_batch(seq2seq_inp) 
                if tf.is_tensor(prediction):
                        prediction = prediction.numpy()
            elif expert.type == Expert_Type.MLP:
                prediction = expert.predict_batch(mlp_inp)
                if tf.is_tensor(prediction):
                    prediction = prediction.numpy()
                prediction = mlp_conversion_func(prediction)
            prediction_list.append(prediction)
            
        return prediction_list

    def create_new_track(self, batch_nr, idx, measurement):
        """Create a new track with the given measurement in an existing batch at position idx.
            
        Adds a new state at the given position of current_states for each model. 
        If batch_nr > number of batches in current_states -> Create new batch

        Args:
            batch_nr (int):     Batch in which the new state is saved
            idx (int):          Position of new state in batch
            measurement (list): The first measurement of the new track
        """
        # Add new state
        for i in range(len(self.experts)):
            expert = self.experts[i]
            if expert.type == Expert_Type.RNN:
                # Create new entry for RNN model
                if len(self.current_states[i]) <= batch_nr:
                    # Create new batch
                    self.current_states[i].append(expert.get_zero_state(self.batch_size))
                else:
                    # Update existing batch
                    # TODO: Understand this shit
                    state_buffers = []
                    for state in self.current_states[i][batch_nr]:
                        state_buffer = np.transpose(state, [1, 0, 2])
                        try:
                            state_buffer[idx] = np.zeros(state_buffer[idx][0].shape, dtype=np.float64)
                        except Exception:
                            logging.error('create_by_id')
                            code.interact(local=dict(globals(), **locals()))
                        state_buffer = np.transpose(state_buffer, [1, 0, 2])
                        state_buffers.append(state_buffer)
                    self.current_states[i][batch_nr] = state_buffers
            elif expert.type == Expert_Type.KF:
                # Create new entry for KF model
                if len(self.current_states[i]) <= batch_nr:
                    # Create new batch
                    self.current_states[i].append(expert.get_zero_state(self.batch_size))

                # Update existing batch
                if isinstance(expert, CV_Model):
                    self.current_states[i][batch_nr][idx] = CV_State(measurement, **expert.default_state_options)
                elif isinstance(expert, CA_Model):
                    self.current_states[i][batch_nr][idx] = CA_State(measurement, **expert.default_state_options)
                else:
                    logging.error("Track creation for expert not implemented!")
            elif expert.type == Expert_Type.MLP:
                # Create new entry for MLP model
                if len(self.current_states[i]) <= batch_nr:
                    # Create new batch
                    self.current_states[i].append(expert.get_zero_state(self.batch_size))
                else:
                    # Update existing batch
                    self.current_states[i][batch_nr][idx] = expert.build_new_state(measurement)
            else:
                logging.error("Track creation for expert not implemented!")

    def predict_all(self, inputs, batch_nr):
        """Predict next state of all models.

        Args:
            inputs (np.array): A batch of inputs (measurements) to the predictors
            batch_nr (int):    The number of the batch to predict

        Returns: 
            A numpy array of all predictions
        """
        all_predictions = []
        for i in range(len(self.experts)):
            expert = self.experts[i]
            if expert.type == Expert_Type.RNN:
                # Predict all tracks of batch with RNN model
                prediction, new_state = expert.predict(inputs, self.current_states[i][batch_nr])
                self.current_states[i][batch_nr] = new_state
                all_predictions.append(prediction)
            elif expert.type == Expert_Type.KF:
                # Predict all tracks of batch with CV model
                prediction = []
                for j in range(len(self.current_states[i][batch_nr])):
                    # update
                    expert.update(self.current_states[i][batch_nr][j], inputs[j])
                    # predict
                    expert.predict(self.current_states[i][batch_nr][j])
                    prediction.append([self.current_states[i][batch_nr][j].get_pos().item(0), 
                                       self.current_states[i][batch_nr][j].get_pos().item(1)])
                all_predictions.append(prediction)
            elif expert.type == Expert_Type.MLP:
                # Predict all tracks of batch with MLP model
                prediction = []
                for j in range(self.current_states[i][batch_nr].shape[0]):
                    expert.update_state(self.current_states[i][batch_nr][j], inputs[j])
                prediction = expert.predict(self.current_states[i][batch_nr])
                all_predictions.append(prediction)
            else:
                logging.error("Track creation for expert not implemented!")
        
        return np.array(all_predictions)

    def get_prediction_mask(self, batch_nr, batch_size=64):
        """Return a mask for predictions of experts.

        Not all expert predictions are valid all the time.
        Right now the MLP is not able to predict values before a certain timestep of the track.

        These values should be masked out.

        Args:
            batch_nr (int):    The number of the batch to predict
            batch_size (int):  The size of the batch (only needed for KF and RNN)

        Returns:
            A list of mask values
        """
        mask = []
        for i in range(len(self.experts)):
            expert = self.experts[i]
            if expert.type == Expert_Type.MLP:
                zero_vals = self.current_states[i][batch_nr] != 0
                all_zero = np.all(zero_vals, axis=1)
                mask.append(all_zero)
            else:
                mask.append(np.ones([batch_size]))
        return np.array(mask)

    def get_n_experts(self):
        """Return n_experts."""
        return self.n_experts

    def get_expert_names(self):
        """Return list of names."""
        return [expert.name for expert in self.experts]

    def get_separation_expert_names(self):
        """Return list of names."""
        return [expert.name for expert in self.separation_experts]

    def get_expert_types(self):
        """Return a list of expert types."""
        return [expert.type for expert in self.experts]

    def is_type_in_experts(self, expert_type):
        """Check if at least one expert is of given type.

        Args: 
            expert_type (Expert_Type): The expert type

        Returns:
            Boolean
        """
        assert(isinstance(expert_type, Expert_Type))
        
        for expert in self.experts:
            if expert.type == expert_type:
                return True 
        return False