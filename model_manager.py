import logging
import tensorflow as tf
import numpy as np
import code  # code.interact(local=dict(globals(), **locals()))

from tensorflow.keras import backend as K

tf.keras.backend.set_floatx('float64')

from rnn_model import RNN_Model


class ModelManager(object):
    def __init__(self, global_config, data_source):
        """
            @variable experts:
            @variable gating_network:
            @varibale weighting_function:
            @variable current_ids:      Stores a global track id for each entry in the batches
                                        current_ids[batch_nr, idx] = global_track_id
            @variable current_inputs:   Stores the next inputs (measurements) for each track id.
                                        Maps the input to a global track id.
                                        current_ids[global_track_id] = [x_in, y_in]
            @variable current_is_alive: Stores if a track is alive.
                                        Maps the alive status to a global track id.
                                        current_is_alive[global_track_id] = [True or False]
            @variable current_states:   Stores the internal states of all tracks for each model.
                                        State structure may vary for each type of expert.
                                        current_states[batch_nr] = [Batch States[...]]     
            @variable current_batches:  Maps a (batch_nr, idx) tuple to each global track id   
            @variable current_free_entries:     Set of all dead free entries in batches                   
        """
        # TODO what variables do we need?
        self.global_config = global_config
        # TODO List of models
        self.rnn_model = RNN_Model(self.global_config, data_source)
        self.rnn_model.rnn_model.reset_states()
        # TODO List of list of states -> Each model has its own list of current states (= particles)
        self.current_states = []  # stored as numpy array for easier access
        self.current_inputs = {}
        self.current_inputs[-1] = [0.0, 0.0]
        self.current_ids = []
        self.current_is_alive = dict()
        self.current_is_alive[-1] = False
        self.current_batches = dict()
        self.current_free_entries = set()

    # TODO create train, test and evaluate functions for single-target tracking

    # TODO can this function be used generally for predict all models?
    def predict_all(self):
        prediction_dict = {}
        for batch_nr in range(len(self.current_states)):
            # state_statetype_first = np.transpose(self.current_states[batch_nr], [1,0,2,3])
            # state_tuple = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_statetype_first[0], state_statetype_first[1])
            
            # get global ids in batch
            global_track_ids = self.current_ids[batch_nr]
            # Get current inputs
            inputs = [self.current_inputs.get(id) for id in global_track_ids]
            # Delte input for dead tracks and save alive tracks of this batch
            alive_tracks = []
            for i in range(len(global_track_ids)):
                if self.current_is_alive.get(global_track_ids[i]):
                    alive_tracks.append(i)
                else:
                    inputs[i] = [0.0, 0.0]
                

            # Predict all 
            prediction, new_state = self.rnn_model.predict(inputs, self.current_states[batch_nr])
            # state_batch_first = np.transpose(self.current_states[batch_nr], [1,0,2,3])
            self.current_states[batch_nr] = new_state

            for i in alive_tracks:
                prediction_dict[global_track_ids[i]] = prediction[i]
        return prediction_dict

    # These are the functions for multi-target tracking
    def update_by_id(self, global_track_id, measurement):
        self.current_inputs[global_track_id] = measurement

    def delete_by_id(self, global_track_id):
        self.current_is_alive[global_track_id] = False
        # Free the entry in this batch if overwriting is activated
        if self.global_config['overwriting_activated']:
            self.current_free_entries.add(self.current_batches.get(global_track_id))

    def create_by_id(self, global_track_id, measurement):
        # Add new track to existing batch if possible
        if len(self.current_free_entries) > 0:
            (batch_nr, idx) = self.current_free_entries.pop()
            self.current_is_alive[global_track_id] = True
            self.current_inputs[global_track_id] = measurement
            self.current_ids[batch_nr][idx] = global_track_id
            self.current_batches[global_track_id] = (batch_nr, idx)
            state_buffers = []
            for state in self.current_states[batch_nr]:
                state_buffer = np.transpose(state, [1, 0, 2])
                try:
                    state_buffer[idx] = np.zeros(state_buffer[idx][0].shape, dtype=np.float64)
                except Exception:
                    logging.error('create_by_id')
                    code.interact(local=dict(globals(), **locals()))
                state_buffer = np.transpose(state_buffer, [1, 0, 2])
                state_buffers.append(state_buffer)
            self.current_states[batch_nr] = state_buffers
            # TODO: Adjust warnings
            if self.global_config['overwriting_activated'] and self.global_config['highest_id'] > idx + self.global_config['batch_size'] * batch_nr and \
                    not self.global_config['state_overwriting_started']:
                self.global_config['state_overwriting_started'] = True
                logging.warning('state_overwriting_started at timestep ' + str(self.global_config['current_time_step']))
                # code.interact(local=dict(globals(), **locals()))
        else:
            # create new batch
            self.current_states.append(self.rnn_model.get_zero_state())
            self.current_ids.append(-np.ones([self.global_config['batch_size']], dtype=np.int32))
            batch_nr = len(self.current_ids)-1
            # declare all entries in new batch as free
            for i in range(self.global_config['batch_size']):
                self.current_free_entries.add((batch_nr, i))
            # fill the first element of the new entry
            self.current_is_alive[global_track_id] = True
            self.current_ids[-1][0] = global_track_id
            self.current_inputs[global_track_id] = measurement
            self.current_batches[global_track_id] = (batch_nr, 0)
            self.current_free_entries.remove((batch_nr, 0))
            if len(self.current_states) > 1:
                logging.info('batch ' + str(len(self.current_states)) + ' is constructed now at timestep' + str(self.global_config['current_time_step']) + '!')
                # code.interact(local=dict(globals(), **locals()))
