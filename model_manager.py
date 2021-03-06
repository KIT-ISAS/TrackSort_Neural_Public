import logging
import tensorflow as tf
import numpy as np
import code  # code.interact(local=dict(globals(), **locals()))

from tensorflow.keras import backend as K

tf.keras.backend.set_floatx('float64')

from model import Model


class ModelManager(object):
    def __init__(self, global_config, data_source, trackManager):
        self.global_config = global_config
        self.model = Model(self.global_config, data_source)
        self.zero_state = self.model.get_zero_state()
        self.model.rnn_model.reset_states()
        self.current_states = []  # stored as numpy array for easier access
        self.current_inputs = []
        self.current_ids = []
        self.current_is_alive = []

        self.trackManager = trackManager

    def predict_all(self):
        prediction_dict = {}
        variances_dict = {}

        for batch_nr in range(len(self.current_states)):

            if self.global_config['mc_dropout']:
                global_track_ids = self.current_ids[batch_nr]
                track_measurement_history = [self.trackManager.tracks[global_track_id].measurements if global_track_id >= 0 else []
                              for global_track_id in global_track_ids  ]
            else:
                track_measurement_history = None

            prediction, new_state, variances = self.model.predict(self.current_inputs[batch_nr],
                                                                  self.current_states[batch_nr],
                                                                  track_measurement_history=track_measurement_history)

            self.current_states[batch_nr] = new_state
            for idx in range(len(self.current_is_alive[batch_nr])):
                if self.current_is_alive[batch_nr][idx]:
                    prediction_dict[self.current_ids[batch_nr][idx]] = prediction[idx]
                    if variances is not None:
                        variances_dict[self.current_ids[batch_nr][idx]] = variances[idx]

        if len(variances_dict) == 0:
            variances_dict = None

        return prediction_dict, variances_dict

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
                            logging.error('create_by_id')
                            code.interact(local=dict(globals(), **locals()))
                        state_buffer = np.transpose(state_buffer, [1, 0, 2])
                        state_buffers.append(state_buffer)
                    self.current_states[batch_nr] = state_buffers
                    if self.global_config['overwriting_activated'] and self.global_config['highest_id'] > idx + self.global_config['batch_size'] * batch_nr and \
                            not self.global_config['state_overwriting_started']:
                        self.global_config['state_overwriting_started'] = True
                        logging.warning('state_overwriting_started at timestep ' + str(self.global_config['current_time_step']))
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
            logging.info('batch ' + str(len(self.current_states)) + ' is constructed now at timestep' + str(self.global_config['current_time_step']) + '!')
            # code.interact(local=dict(globals(), **locals()))
