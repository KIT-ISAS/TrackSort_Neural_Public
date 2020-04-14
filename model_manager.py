"""Model Manager.

Todo:
    * Add train and test method
    * Convert np representation to tensor representation for mixture of experts
"""

import logging
import tensorflow as tf
import numpy as np
import code  # code.interact(local=dict(globals(), **locals()))
import time
import datetime

from tensorflow.keras import backend as K

from expert_manager import Expert_Manager
from weighting_function import weighting_function
from ensemble import Simple_Ensemble
from evaluation_functions import *

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

    def __init__(self, model_config, is_loaded, num_time_steps, overwriting_activated=True):
        """Initialize a model manager.

        Creates the expert manager and gating network.
        Initializes attributes.

        Args:
            model_config (dict):  The json tree containing all information about the experts, gating network and weighting function
            num_time_steps (int): The number of timesteps in the longest track
            overwriting_activated (Boolean): Should expired tracks in batches be overwritten with new tracks
        """
        # The manager of all the models
        self.expert_manager = Expert_Manager(model_config.get('experts'), is_loaded, model_config.get('model_path'), 
                                             model_config.get('batch_size'), num_time_steps)
        # The gating network that calculates all weights
        self.create_gating_network(model_config.get('gating'))
        self.overwriting_activated = overwriting_activated
        self.batch_size = model_config.get('batch_size')
        self.num_time_steps = num_time_steps

        self.current_inputs = {}
        self.current_inputs[-1] = [0.0, 0.0]
        self.current_ids = []
        self.current_is_alive = dict()
        self.current_is_alive[-1] = False
        self.current_batches = dict()
        self.current_free_entries = set()

        self.mask_value = np.array([.0, .0])

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

    def train_models(self, mlp_conversion_func, 
                    seq2seq_dataset_train = None, seq2seq_dataset_test = None,
                    mlp_dataset_train = None, mlp_dataset_test = None,
                    num_train_epochs = 1000, evaluate_every_n_epochs = 20,
                    improvement_break_condition = 0.001, lr_decay_after_epochs = 100, lr_decay = 0.1):
        """Train all experts and the gating network.

        The training information of each model should be provided in the configuration json.

        Args:
            mlp_conversion_func:            Function to convert MLP format to track format
            **_dataset_train (dict):        All training samples in the correct format for various models
            **_dataset_test (dict):         All testing samples for evaluating the trained models
            num_train_epochs (int):         Number of epochs for training
            evaluate_every_n_epochs (int):  Evaluate the trained models every n epochs on the test data
            improvement_break_condition (double): Break training if test loss on every expert does not improve by more than this value.
            lr_decay_after_epochs (int):    Decrease the learning rate of certain models (RNNs) after x epochs
            lr_decay (double):              Learning rate decrease (multiplicative). Choose values < 1 to increase accuracy.
        """
        train_losses = []
        
        # Define a loss function: MSE
        loss_object = tf.keras.losses.MeanSquaredError()
        mae_object = tf.keras.losses.MeanAbsoluteError()
        # Mask value for keras
        k_mask_value = K.variable(self.mask_value, dtype=tf.float64)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Create a tensorboard folder and writer for every expert
        expert_names = self.expert_manager.get_expert_names()
        train_log_dirs = []
        test_log_dirs = []
        train_summary_writers = []
        test_summary_writers = []
        # Define a metric for calculating the train loss: MEAN
        train_losses = []
        train_maes = []
        test_losses = []
        test_maes = []
        test_mlp_losses = []
        old_test_losses = []
        for name in expert_names:
            train_log_dirs.append('logs/gradient_tape/' + current_time + '/' + name +'/train')
            test_log_dirs.append('logs/gradient_tape/' + current_time + '/' + name + '/test')
            train_summary_writers.append(tf.summary.create_file_writer(train_log_dirs[-1]))
            test_summary_writers.append(tf.summary.create_file_writer(test_log_dirs[-1]))
            train_losses.append(tf.keras.metrics.Mean('train_loss', dtype=tf.float32))
            test_losses.append(tf.keras.metrics.Mean('test_loss', dtype=tf.float32))
            train_maes.append(tf.keras.metrics.Mean('train_mae', dtype=tf.float32))
            test_maes.append(tf.keras.metrics.Mean('test_mae', dtype=tf.float32))
            test_mlp_losses.append(tf.keras.metrics.Mean('test_mlp_loss', dtype=tf.float32))
            old_test_losses.append(1)


        for epoch in range(num_train_epochs):
            start_time = time.time()

            # learning rate decay after x epochs
            if (epoch + 1) % lr_decay_after_epochs == 0:
                logging.info("Decreasing learning rate...")
                self.expert_manager.change_learning_rate(lr_decay)
            
            seq2seq_iter = iter(seq2seq_dataset_train)
            mlp_iter = iter(mlp_dataset_train)

            for (seq2seq_inp, seq2seq_target) in seq2seq_iter:
                (mlp_inp, mlp_target) = next(mlp_iter)
                # Train experts on a batch
                predictions = self.expert_manager.train_batch(mlp_conversion_func, seq2seq_inp, seq2seq_target, mlp_inp, mlp_target)
                # Create a mask for end of tracks and for beginning of tracks (MLP)
                masks = self.expert_manager.get_masks(mlp_conversion_func, k_mask_value, seq2seq_target, mlp_target)
                # Calculate loss for all models
                for i in range(len(predictions)):
                    loss = loss_object(seq2seq_target, predictions[i], sample_weight = masks[i])
                    mae = mae_object(seq2seq_target, predictions[i], sample_weight = masks[i])
                    train_losses[i](loss)
                    train_maes[i](mae)
                
            
            for i in range(len(train_summary_writers)):
                with train_summary_writers[i].as_default():
                    tf.summary.scalar('loss', train_losses[i].result(), step=epoch)
                    tf.summary.scalar('mae', train_maes[i].result(), step=epoch)

            template = 'Epoch {}, Train Losses: {}, Train MAEs: {}'
            logging.info((template.format(epoch+1,
                                          [train_loss.result().numpy() for train_loss in train_losses],
                                          [train_mae.result().numpy() for train_mae in train_maes])))

            # Reset metrics every epoch
            for i in range(len(train_losses)):
                train_losses[i].reset_states()
                train_maes[i].reset_states()

            # Run trained models on the test set every n epochs
            if (epoch + 1) % evaluate_every_n_epochs == 0 \
                    or (epoch + 1) == num_train_epochs:
                seq2seq_iter = iter(seq2seq_dataset_test)
                mlp_iter = iter(mlp_dataset_test)

                for (seq2seq_inp, seq2seq_target) in seq2seq_iter:
                    (mlp_inp, mlp_target) = next(mlp_iter)
                    # Test experts on a batch
                    predictions = self.expert_manager.test_batch(mlp_conversion_func, seq2seq_inp, mlp_inp)
                    masks = self.expert_manager.get_masks(mlp_conversion_func, k_mask_value, seq2seq_target, mlp_target)
                    # MLP mask to compare MLP with KF/RNN
                    mlp_mask = K.all(K.equal(mlp_conversion_func(mlp_target), k_mask_value), axis=-1)
                    mlp_mask = 1 - K.cast(mlp_mask, tf.float64)
                    # Calculate loss for all models
                    #TODO: Implement MLP loss as loss if all predictors would have MLP mask
                    for i in range(len(predictions)):
                        loss = loss_object(seq2seq_target, predictions[i], sample_weight = masks[i])
                        mlp_loss = loss_object(seq2seq_target, predictions[i], sample_weight = mlp_mask)
                        mae = mae_object(seq2seq_target, predictions[i], sample_weight = masks[i])
                        test_losses[i](loss)
                        test_maes[i](mae)
                        test_mlp_losses[i](mlp_loss)
                    
                for i in range(len(test_summary_writers)):
                    with test_summary_writers[i].as_default():
                        tf.summary.scalar('loss', test_losses[i].result(), step=epoch)
                        tf.summary.scalar('mlp loss', test_mlp_losses[i].result(), step=epoch)
                        tf.summary.scalar('mae', test_maes[i].result(), step=epoch)

                template = 'Epoch {}, Test Losses: {}, Test MAEs: {}, Test MLP Losses: {}'
                logging.info((template.format(epoch+1,
                                            [test_loss.result().numpy() for test_loss in test_losses],
                                            [test_mae.result().numpy() for test_mae in test_maes],
                                            [test_mlp_loss.result().numpy() for test_mlp_loss in test_mlp_losses])))

                # Check testing break condition
                break_condition = True
                for i in range(len(old_test_losses)):
                    # If the percentage improvement of one expert is higher than the threshold, we do not break.
                    if (old_test_losses[i]-test_losses[i].result().numpy())/old_test_losses[i] >= improvement_break_condition:
                        break_condition = False
                    # Update old test losses
                    old_test_losses[i] = test_losses[i].result().numpy()
                # Break if there was no improvement in all experts
                if break_condition:
                    log_string = "Break training because improvement of experts on all tests was lower than {}%.".format(
                                    improvement_break_condition*100)
                    logging.info(log_string)
                    break 

        # Save all models
        self.expert_manager.save_models()

    def test_models(self, mlp_conversion_func, 
                    seq2seq_dataset_test = None, mlp_dataset_test = None):
        """Test model performance on test dataset and create evaluations.

        Args:
            mlp_conversion_func: Function to convert MLP format to track format
            **_dataset_test (tf.Tensor): Batches of test data
        """
         # Define a loss function: MSE
        k_mask_value = K.variable(self.mask_value, dtype=tf.float64)
        loss_object = tf.keras.losses.MeanSquaredError()
        mae_object = tf.keras.losses.MeanAbsoluteError()
        seq2seq_iter = iter(seq2seq_dataset_test)
        mlp_iter = iter(mlp_dataset_test)

        all_losses = []; all_mlp_losses = []; all_maes = []

        for (seq2seq_inp, seq2seq_target) in seq2seq_iter:
            (mlp_inp, mlp_target) = next(mlp_iter)
            # Test experts on a batch
            predictions = self.expert_manager.test_batch(mlp_conversion_func, seq2seq_inp, mlp_inp)
            masks = self.expert_manager.get_masks(mlp_conversion_func, k_mask_value, seq2seq_target, mlp_target)
            # MLP mask to compare MLP with KF/RNN
            mlp_mask = K.all(K.equal(mlp_conversion_func(mlp_target), k_mask_value), axis=-1)
            mlp_mask = 1 - K.cast(mlp_mask, tf.float64)
            # Calculate loss for all models
            losses = []; mlp_losses = []; maes = []
            for i in range(len(predictions)):
                losses.append(loss_object(seq2seq_target, predictions[i], sample_weight = masks[i]))
                mlp_losses.append(loss_object(seq2seq_target, predictions[i], sample_weight = mlp_mask))
                maes.append(mae_object(seq2seq_target, predictions[i], sample_weight = masks[i]))
            all_losses.append(losses)
            all_mlp_losses.append(mlp_losses)
            all_maes.append(maes)
            
        """
        predictions = []
        targets = []
        n_batches = 0
        for (batch_n, (inp, target)) in enumerate(dataset_test):
            prediction = self.expert_manager.test_batch(inp)
            predictions.append(np.array(prediction))
            targets.append(target)
            n_batches += 1
        
        # Reshape the data to get rid of the batches.
        np_predictions = np.zeros([self.expert_manager.get_n_experts(), n_batches*self.batch_size, self.num_time_steps, 2])
        np_targets = np.zeros([n_batches*self.batch_size, self.num_time_steps, 2])
        for i in range(n_batches):
            np_predictions[:, i*self.batch_size:(i+1)*self.batch_size, :, :] = np.array(predictions[i])
            np_targets[i*self.batch_size:(i+1)*self.batch_size, :, :] = targets[i].numpy()

        calculate_mse(np_targets, np_predictions, self.mask_value)
        find_worst_predictions(np_targets, np_predictions, self.mask_value)
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
            prediction_mask = self.expert_manager.get_prediction_mask(batch_nr, batch_size=len(global_track_ids))
            # Gating
            # TODO: Implement MLP masking in gating network!
            #weights = np.array([[1], [0]])
            weights = self.gating_network.get_masked_weights(prediction_mask)

            # Weighting
            prediction = weighting_function(all_predictions, weights)

            for i in alive_tracks:
                # TODO cleaner solution
                prediction_dict[global_track_ids[i]] = prediction[i]
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
        if self.overwriting_activated:
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
            self.current_ids.append(-np.ones(self.batch_size, dtype=np.int32))
            batch_nr = len(self.current_ids)-1
            idx = 0
            # declare all entries in new batch as free
            for i in range(self.batch_size):
                self.current_free_entries.add((batch_nr, i))
            # remove first entry from free list
            self.current_free_entries.remove((batch_nr, 0))
            if batch_nr > 0:
                logging.info('batch ' + str(batch_nr) + ' is constructed!')
                # code.interact(local=dict(globals(), **locals()))

        # Fill free entry with new track
        self.current_is_alive[global_track_id] = True
        self.current_inputs[global_track_id] = measurement
        self.current_ids[batch_nr][idx] = global_track_id
        self.current_batches[global_track_id] = (batch_nr, idx)
        self.expert_manager.create_new_track(batch_nr, idx, measurement)
