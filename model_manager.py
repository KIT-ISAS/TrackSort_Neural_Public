import logging
import tensorflow as tf
import numpy as np
import code  # code.interact(local=dict(globals(), **locals()))

#from tensorflow.keras import backend as K

from expert_manager import Expert_Manager
from weighting_function import weighting_function

#tf.keras.backend.set_floatx('float64')


class ModelManager(object):
    def __init__(self, global_config, data_source, model_config):
        """
            @param global_config
            @param data_source          The data source object
            @param model_config         The json tree containing all information about the experts, gating network and wighting function

            @variable expert_manager:   Object handling all experts (creation, states, prediction, ...)
            @variable gating_network:

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

        # The manager of all the models
        self.expert_manager = Expert_Manager(global_config, model_config.get('experts'), data_source)
        
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
        """
            Predict the next position with all experts.
            Predicts a whole batch at a time.

            @return predictions for all alive tracks from all models
        """
        prediction_dict = {}
        for batch_nr in range(len(self.current_ids)):
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
                
            # Predict the next state for all models
            all_predictions = self.expert_manager.predict_all(inputs, batch_nr)

            # Gating
            #weights = np.array([[1], [0]])
            weights = (1/all_predictions.shape[0]) * np.ones([all_predictions.shape[0],1])

            # Weighting
            prediction = weighting_function(all_predictions, weights)

            for i in alive_tracks:
                # TODO cleaner solution
                prediction_dict[global_track_ids[i]] = np.array([prediction[i,0,0], prediction[i,1,0]])
        return prediction_dict

    # These are the functions for multi-target tracking
    def update_by_id(self, global_track_id, measurement):
        """
            Update a track with a new measurement for all models.    

            @param gobal_track_id:  The id of the track from which the new measurement stems
            @param measurement:     The new measurement [x_in, y_in]
        """
        self.current_inputs[global_track_id] = measurement

    def delete_by_id(self, global_track_id):
        """
            Delete a track

            @param gobal_track_id:  The id of the track
        """
        self.current_is_alive[global_track_id] = False
        # Free the entry in this batch if overwriting is activated
        if self.global_config['overwriting_activated']:
            self.current_free_entries.add(self.current_batches.get(global_track_id))

    def create_by_id(self, global_track_id, measurement):
        """
            Create a new track with a new measurement for all models. 
            Creates a new batch of tracks if there is no free space in the existing batches.   

            @param gobal_track_id:  The id of the track from which the new measurement stems
            @param measurement:     The new measurement [x_in, y_in]
        """
        batch_nr = 0
        idx = 0
        # Add new track to existing batch if possible
        if len(self.current_free_entries) > 0:
            # Get the next free entry (unordered!)
            (batch_nr, idx) = self.current_free_entries.pop()
        else:
            # create new batch
            self.current_ids.append(-np.ones([self.global_config['batch_size']], dtype=np.int32))
            batch_nr = len(self.current_ids)-1
            idx = 0
            # declare all entries in new batch as free
            for i in range(self.global_config['batch_size']):
                self.current_free_entries.add((batch_nr, i))
            # remove first entry from free list
            self.current_free_entries.remove((batch_nr, 0))
            if batch_nr > 0:
                logging.info('batch ' + str(batch_nr) + ' is constructed now at timestep ' + str(self.global_config['current_time_step']) + '!')
                # code.interact(local=dict(globals(), **locals()))

        # Fill free entry with new track
        self.current_is_alive[global_track_id] = True
        self.current_inputs[global_track_id] = measurement
        self.current_ids[batch_nr][idx] = global_track_id
        self.current_batches[global_track_id] = (batch_nr, idx)
        self.expert_manager.create_new_track(batch_nr, idx, measurement)
