"""Expert Manager.

Todo:
    * Delete global_config
    * Add train and test method
    * Move data_source to training and test methods
    * Convert np representation to tensor representation for mixture of experts
"""

import logging
#import tensorflow as tf
import numpy as np
import code

#from tensorflow.keras import backend as K

#tf.keras.backend.set_floatx('float64')

from rnn_model import RNN_Model
from cv_model import CV_Model, CV_State

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
    def __init__(self, global_config, expert_config, data_source):
        """Initialize an expert manager.

        Creates the expert models.
        Initializes attributes.

        Args:
            expert_config (dict): The configuration dictionary of all experts
            data_source:   TODO remove
            global_config: TODO remove
        """
        self.expert_config = expert_config
        # TODO: Remove global config and data source
        self.global_config = global_config
        self.data_source = data_source
        # List of list of states -> Each model has its own list of current states (= particles)
        self.current_states = []
        self.create_models(expert_config)
        self.n_experts = len(self.experts)

    def create_models(self):
        """Create list of experts.

        Creat experts based on self.expert_cofig.
        Add empty list to list of states for each expert.
        """
        self.experts = []

        for expert_name in self.expert_config:
            expert = self.expert_config.get(expert_name)
            expert_type = expert.get("type")
            if expert_type == 'RNN':
                # Create RNN model
                rnn_model = RNN_Model(self.global_config, expert.get("options"), self.data_source)
                rnn_model.rnn_model.reset_states()
                self.experts.append(rnn_model)
                self.current_states.append([])
            elif expert_type == 'KalmanFilter':
                # Create Kalman filter model
                sub_type = expert.get("sub_type")
                if sub_type == 'CV':
                    # Create constant velocity model
                    cv_model = CV_Model(**expert.get("model_options"), default_state_options=expert.get("state_options"))
                    self.experts.append(cv_model)
                    self.current_states.append([])
                else:
                    logging.warning("Kalman filter subtype " + sub_type + " not supported. Will not create model.") 
            else:
                logging.warning("Expert type " + expert_type + " not supported. Will not create model.")

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
            elif isinstance(expert, CV_Model):
                # Create new entry for CV model
                if len(self.current_states[i]) <= batch_nr:
                    # Create new batch
                    self.current_states[i].append(expert.get_zero_state(self.global_config.get("batch_size")))

                # Update existing batch
                self.current_states[i][batch_nr][idx] = CV_State(measurement, **expert.default_state_options)
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
            elif isinstance(expert, CV_Model):
                # Predict all tracks of batch with CV model
                #TODO: Is batch wise prediction possible? --> Calculation time is very high
                
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

