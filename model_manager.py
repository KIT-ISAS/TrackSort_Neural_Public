"""Model Manager.

Todo:
    * Delete global_config
    * Add train and test method
    * Move data_source to training and test methods
    * Convert np representation to tensor representation for mixture of experts
"""

import logging
import tensorflow as tf
import numpy as np
import code  # code.interact(local=dict(globals(), **locals()))
import time

#from tensorflow.keras import backend as K

from expert_manager import Expert_Manager
from weighting_function import weighting_function
from ensemble import Simple_Ensemble

#tf.keras.backend.set_floatx('float64')


class ModelManager(object):
    """The model manager handles all models for single target tracking.

    Handles the experts, the gating network and the weighting of experts.
    Provides train and test methods.
    Handles inputs for experts for tracking of multiple single particles (No data association is done here).

    Attributes:
        expert_manager:     Object handling all experts (creation, states, prediction, ...)
        gating_network:     Object creating the weights for each expert
        current_ids (list):     Stores a global track id for each entry in the batches
                                    current_ids[batch_nr, idx] = global_track_id
        current_inputs (dict):  Stores the next inputs (measurements) for each track id.
                                    Maps the input to a global track id.
                                    current_ids[global_track_id] = [x_in, y_in]
        current_is_alive (dict): Stores if a track is alive.
                                    Maps the alive status to a global track id.
                                    current_is_alive[global_track_id] = [True or False]
        current_batches (dict): Maps a (batch_nr, idx) tuple to each global track id
        current_free_entries (set): Set of all dead free entries in batches
    """

    def __init__(self, global_config, data_source, model_config, is_loaded, num_time_steps):
        """Initialize a model manager.

        Creates the expert manager and gating network.
        Initializes attributes.

        Args:
            global_config (dict): To be removed
            data_source (object): To be moved to training and test functions
            model_config (dict):  The json tree containing all information about the experts, gating network and weighting function
            num_time_steps (int): The number of timesteps in the longest track
        """
        # TODO what variables do we need?
        self.global_config = global_config

        # The manager of all the models
        self.expert_manager = Expert_Manager(global_config, model_config.get('experts'), data_source,
                                             is_loaded, model_config.get('model_path'), model_config.get('batch_size'),
                                             num_time_steps)
        # The gating network that calculates all weights
        self.create_gating_network(model_config.get('gating'))
        
        self.current_inputs = {}
        self.current_inputs[-1] = [0.0, 0.0]
        self.current_ids = []
        self.current_is_alive = dict()
        self.current_is_alive[-1] = False
        self.current_batches = dict()
        self.current_free_entries = set()

    def create_gating_network(self, gating_config):
        """Create the gating network.

        Creates a gating network according to the given config

        Args:
            gating_config (dict): Includes information about type and options of the gating network

        """
        gating_type = gating_config.get("type")
        if gating_type == "Simple_Ensemble":
            self.gating_network = Simple_Ensemble(self.expert_manager.n_experts)
        else:
            raise Exception("Unknown gating type '" + gating_type + "'!")

    # TODO create train, test and evaluate functions for single-target tracking
    def train_models(self, dataset_train, num_train_epochs = 1000, evaluate_every_n_epochs=20):
        """Train all experts and the gating network.

        The training information of each model should be provided in the configuration json.

        Args:
            dataset_train (dict): All training samples in the correct format for various models
            num_train_epochs (int): Number of epochs for training the overall model
        """
        train_losses = []
        for epoch in range(num_train_epochs):
            start_time = time.time()
            # TODO: Implement lr decay
            """
            # learning rate decay after 100 epochs
            if (epoch + 1) % self.global_config['lr_decay_after_epochs'] == 0:
                old_lr = K.get_value(optimizer.lr)
                new_lr = old_lr * self.global_config['lr_decay_factor']
                logging.info("Reducing learning rate from {} to {}.".format(old_lr, new_lr))
                K.set_value(optimizer.lr, new_lr)
            """
            mse_batch = []
            mae_batch = []
            for (batch_n, (inp, target)) in enumerate(dataset_train):
                mse_list, mae_list = self.expert_manager.train_batch(inp, target)
                mse_batch.append(mse_list)
                mae_batch.append(mae_list)
                stop=0

            mean_mse = np.mean(np.array(mse_batch), 0)
            mean_mae = np.mean(np.array(mae_batch), 0)
            total_mse = np.mean(mean_mse)
            total_mae = np.mean(mean_mae)
            stop = 0
            
            train_losses.append([epoch, total_mse, total_mae]) #total_mae * self.data_source.normalization_constant

            log_string = "{}/{}: \t loss={}".format(epoch, num_train_epochs, total_mse)
            end_time = time.time()
            logging.info("Batch trained, time needed: " + str(end_time - start_time))
            # Evaluate
            if (epoch + 1) % evaluate_every_n_epochs == 0 \
                    or (epoch + 1) == num_train_epochs:
                logging.info(log_string)
            """
                test_mse, test_mae = self._evaluate_model(dataset_test, epoch)
                test_losses.append([epoch, test_mse, test_mae * self.data_source.normalization_constant])
            else:
                logging.debug(log_string)
            """
            
        """
        # Visualize loss curve
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        # MSE
        plt.plot(train_losses[:, 0], train_losses[:, 1], c='blue', label="Training MSE")
        plt.plot(test_losses[:, 0], test_losses[:, 1], c='red', label="Test MSE")
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MSE.png'))
        plt.clf()

        # MAE
        plt.plot(train_losses[:, 0], train_losses[:, 2], c='blue', label="Training MAE (not normalized)")
        plt.plot(test_losses[:, 0], test_losses[:, 2], c='red', label="Test MAE (not normalized)")
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig(os.path.join(self.global_config['diagrams_path'], 'MAE.png'))
        plt.clf()
        """

    def load_models(self, model_path):
        """Load experts and gating network from path.

        Args:
            model_path (string): Path to saved models

        Todo:
            Implement load function
        """
        pass

    def predict_all(self):
        """Predict the next position with all experts.
        
        Predict a whole batch of particles at a time.

        Returns:
            Predictions for all alive tracks from all models
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
            weights = self.gating_network.get_weights()

            # Weighting
            prediction = weighting_function(all_predictions, weights)

            for i in alive_tracks:
                # TODO cleaner solution
                prediction_dict[global_track_ids[i]] = np.array([prediction[i,0,0], prediction[i,1,0]])
        return prediction_dict

    # These are the functions for multi-target tracking
    def update_by_id(self, global_track_id, measurement):
        """Update a track with a new measurement for all models.

        Args:
            gobal_track_id (int): The id of the track from which the new measurement stems
            measurement (list):   The new measurement [x_in, y_in]
        """
        self.current_inputs[global_track_id] = measurement

    def delete_by_id(self, global_track_id):
        """Delete a track.

        Args:
            gobal_track_id (int):  The id of the track to delte
        """
        self.current_is_alive[global_track_id] = False
        # Free the entry in this batch if overwriting is activated
        if self.global_config['overwriting_activated']:
            self.current_free_entries.add(self.current_batches.get(global_track_id))

    def create_by_id(self, global_track_id, measurement):
        """Create a new track with a new measurement for all models.
            
        Creates a new batch of tracks if there is no free space in the existing batches.   

        Args:
            gobal_track_id (int):  The id of the track from which the new measurement stems
            measurement (list):    The new measurement [x_in, y_in]
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
