import logging
#import tensorflow as tf
import numpy as np
import code

#from tensorflow.keras import backend as K

#tf.keras.backend.set_floatx('float64')

from rnn_model import RNN_Model
from cv_model import CV_Model, CV_State

class Expert_Manager(object):
    def __init__(self, global_config, expert_config, data_source):
        """
            @param expert_config: The configuration dictionary of all experts
            @param data_source: TODO remove
            @param global_config: TODO remove
        """
        self.expert_config = expert_config
        # TODO: Remove global config and data source
        self.global_config = global_config
        self.data_source = data_source
        # List of list of states -> Each model has its own list of current states (= particles)
        self.current_states = []
        self.create_models(expert_config)
        self.n_experts = len(self.experts)

    def create_models(self, expert_config):
        """
            Create list of experts

            @param expert_config:   Config dict that contains all options to all experts to create
        """
        self.experts = []

        for expert_name in expert_config:
            expert = expert_config.get(expert_name)
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
        """
            Create a new track with the given measurement in an existing batch at position idx.
            Adds a new state at the given position of current_states for each model. 
            If batch_nr > number of batches in current_states -> Create new batch

            @param batch_nr:    Batch in which the new state is saved
            @param idx:         Position of new state in batch
            @param measurement: The first measurement of the new track #TODO remove measurement!!!
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
        """
            predict next state of all models

            @param inputs: A batch of inputs (measurements) to the predictors

            @return: A list of all predictions
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

