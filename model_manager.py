import math, copy
import tensorflow as tf
import numpy as np
import code  # code.interact(local=dict(globals(), **locals()))

from tensorflow.keras import backend as K

tf.keras.backend.set_floatx('float64')

from model import Model


class ModelManager(object):
    def __init__(self, global_config, data_source):
        self.global_config = global_config
        self.model = Model(self.global_config, data_source)
        self.zero_state = self.model.get_zero_state()
        self.model.rnn_model.reset_states()
        self.current_states = []  # stored as numpy array for easier access
        self.current_inputs = []
        self.current_ids = []
        self.current_is_alive = []

    def predict_all(self):
        prediction_dict = {}
        for batch_nr in range(len(self.current_states)):
            # if self.global_config['verbos'] >= 2: print('in predict_all')
            # code.interact(local=dict(globals(), **locals()))
            # state_statetype_first = np.transpose(self.current_states[batch_nr], [1,0,2,3])
            # state_tuple = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_statetype_first[0], state_statetype_first[1])
            prediction, new_state = self.model.predict(self.current_inputs[batch_nr], self.current_states[batch_nr])
            # state_batch_first = np.transpose(self.current_states[batch_nr], [1,0,2,3])
            self.current_states[batch_nr] = new_state
            for idx in range(len(self.current_is_alive[batch_nr])):
                if self.current_is_alive[batch_nr][idx]:
                    prediction_dict[self.current_ids[batch_nr][idx]] = prediction[idx]
        return prediction_dict

    def update_by_id(self, global_track_id, measurement):
        for batch_nr in range(len(self.current_ids)):
            for idx in range(len(self.current_ids[batch_nr])):
                if self.current_ids[batch_nr][idx] == global_track_id:
                    self.current_inputs[batch_nr][idx] = measurement

    def delete_by_id(self, global_track_id):
        for batch_nr in range(len(self.current_ids)):
            for idx in range(len(self.current_ids[batch_nr])):
                if self.current_ids[batch_nr][idx] == global_track_id:
                    self.current_is_alive[batch_nr][idx] = False
                    self.current_ids[batch_nr][idx] = -1

    def create_by_id(self, global_track_id, measurement):
        for batch_nr in range(len(self.current_is_alive)):
            for idx in range(len(self.current_is_alive[batch_nr])):
                if (self.global_config['overwriting_activated'] and not self.current_is_alive[batch_nr][idx]) or \
                        (not self.global_config['overwriting_activated'] and self.global_config['highest_id'] == idx + self.global_config['batch_size'] * batch_nr):
                    self.current_is_alive[batch_nr][idx] = True
                    self.current_inputs[batch_nr][idx] = measurement
                    self.current_ids[batch_nr][idx] = global_track_id
                    state_buffers = []
                    for state in self.current_states[batch_nr]:
                        state_buffer = np.transpose(state, [1, 0, 2])
                        try:
                            state_buffer[idx] = np.zeros(state_buffer[idx][0].shape, dtype=np.float64)
                        except Exception:
                            if self.global_config['verbos'] >= 2: print('create_by_id')
                            code.interact(local=dict(globals(), **locals()))
                        state_buffer = np.transpose(state_buffer, [1, 0, 2])
                        state_buffers.append(state_buffer)
                    self.current_states[batch_nr] = state_buffers
                    if self.global_config['overwriting_activated'] and self.global_config['highest_id'] > idx + self.global_config['batch_size'] * batch_nr and \
                            not self.global_config['state_overwrinting_started']:
                        self.global_config['state_overwriting_started'] = True
                        if self.global_config['verbos'] >= 2: print('state_overwriting_started at timestep ' + str(self.global_config['current_time_step']))
                        # code.interact(local=dict(globals(), **locals()))
                    return
        # create new entry of the lists
        self.current_states.append(self.zero_state)
        self.current_is_alive.append(np.zeros([self.global_config['batch_size']], dtype=bool))
        self.current_ids.append(-np.ones([self.global_config['batch_size']], dtype=np.int32))
        self.current_inputs.append(np.zeros([self.global_config['batch_size'], 2], dtype=np.float64))
        # fill the first element of the new entry
        self.current_is_alive[-1][0] = True
        self.current_ids[-1][0] = global_track_id
        self.current_inputs[-1][0] = measurement
        if len(self.current_states) > 1:
            if self.global_config['verbos'] >= 2: print('batch ' + str(len(self.current_states)) + ' is constructed now at timestep' + str(self.global_config['current_time_step']) + '!')
            # code.interact(local=dict(globals(), **locals()))
