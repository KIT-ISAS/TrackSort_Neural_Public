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
    
    def __init__(self, expert_config, is_loaded, model_path="", batch_size=64, num_time_steps=0):
        """Initialize an expert manager.

        Creates the expert models.
        Initializes attributes.

        Args:
            expert_config (dict): The configuration dictionary of all experts
            is_loaded (Boolean):  True for loading models, False for creating new models
            model_path (String):  The path of the models if is_loaded is True
            batch_size (int):     The batch size of the data
            num_time_steps (int): The number of timesteps in the longest track
        """
        self.expert_config = expert_config
        self.batch_size = batch_size
        # List of list of states -> Each model has its own list of current states (= particles)
        self.current_states = []
        self.experts = []
        self.create_models(is_loaded, model_path, batch_size, num_time_steps)
        self.n_experts = len(self.experts)

    def create_models(self, is_loaded, model_path="", batch_size=64, num_time_steps=0, n_mlp_features = 10):
        """Create list of experts.

        Creat experts based on self.expert_cofig.
        Add empty list to list of states for each expert.
        Load experts from model_path if is_loaded is True.
        Create new experts if is_loaded is False.

        Args:
            is_loaded (Boolean):  True for loading models, False for creating new models
            model_path (String):  The path of the models if is_loaded is True
            batch_size (int):     The batch size of the data
            num_time_steps (int): The number of timesteps in the longest track
            n_mlp_features (int): The number of features for MLP networks
        """
        for expert_name in self.expert_config:
            expert = self.expert_config.get(expert_name)
            expert_type = expert.get("type")
            if expert_type == 'RNN':
                # Create RNN model
                model_path = expert.get("model_path")
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
                    cv_model = CV_Model(expert_name, **expert.get("model_options"), default_state_options=expert.get("state_options"))
                    self.experts.append(cv_model)
                    self.current_states.append([])
                elif sub_type == 'CA':
                    # Create constant velocity model
                    ca_model = CA_Model(expert_name, **expert.get("model_options"), default_state_options=expert.get("state_options"))
                    self.experts.append(ca_model)
                    self.current_states.append([])
                else:
                    logging.warning("Kalman filter subtype " + sub_type + " not supported. Will not create model.") 
            elif expert_type=='MLP':
                model_path = expert.get("model_path")
                mlp_model = MLP_Model(expert_name, model_path, True, expert.get("options"))
                if is_loaded:
                    mlp_model.load_model()
                else:
                    mlp_model.create_model(n_mlp_features)
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
        """Train one batch for all experts.

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
            if isinstance(expert, RNN_Model):
                # Create new entry for RNN model
                if len(self.current_states[i]) <= batch_nr:
                    # Create new batch
                    self.current_states[i].append(expert.get_zero_state())
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
            elif isinstance(expert, KF_Model):
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
            else:
                logging.error("Track creation for expert not implemented!")

    def predict_all(self, inputs, batch_nr):
        """Predict next state of all models.

        Args:
            inputs (np.array): A batch of inputs (measurements) to the predictors

        Returns: 
            A numpy array of all predictions
        """
        all_predictions = []
        for i in range(len(self.experts)):
            expert = self.experts[i]
            if isinstance(expert, RNN_Model):
                # Predict all tracks of batch with RNN model
                prediction, new_state = expert.predict(inputs, self.current_states[i][batch_nr])
                self.current_states[i][batch_nr] = new_state
                all_predictions.append(prediction)
            elif isinstance(expert, KF_Model):
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
                
            else:
                logging.error("Track creation for expert not implemented!")
        
        return np.array(all_predictions)

    def get_n_experts(self):
        """Return n_experts."""
        return self.n_experts

    def get_expert_names(self):
        """Return list of names."""
        return [expert.name for expert in self.experts]