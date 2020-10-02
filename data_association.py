"""Data association class and NN function.

Change log (Please insert your name here if you worked on this file)
    * Created by: Daniel Pollithy
    * Jakob Thumm (jakob.thumm@student.kit.edu) 2.10.2020:    Completed documentation.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import code  # code.interact(local=dict(globals(), **locals()))
import shutil
import math
import os
import logging

from track_manager import TrackManager
from data_manager import FakeDataSet, CsvDataSet

def nearest_neighbour(weight_matrix):
    """Local Nearest neighbour algorithm.

    Args:
        weight_matrix (np.array):   Matrix containing the distance between each (virtual) measurement and (virtual) prediction

    Returns:
        measurement_idxs (list), prediction_idxs (list)
    """
    num_rows = weight_matrix.shape[0]
    num_cols = weight_matrix.shape[1]
    measurement_idxs = []
    prediction_idxs = []
    sorted_idxs = np.argsort(weight_matrix.flatten())
    for idx in range(sorted_idxs.shape[0]):
        row = int(sorted_idxs[idx] / num_cols)
        col = sorted_idxs[idx] - row * num_cols
        if not row in measurement_idxs and not col in prediction_idxs:
            measurement_idxs.append(row)
            prediction_idxs.append(col)
            if len(measurement_idxs) == num_rows or len(prediction_idxs) == num_cols:
                return measurement_idxs, prediction_idxs
    logging.error('something went wrong in local nearest_neighbour!')
    code.interact(local=dict(globals(), **locals()))


class DataAssociation(object):
    def __init__(self, num_timesteps, rotate_columns, visualization_path, visualize, matching_algorithm,
                delta_start_end_phase=0.025, no_change_dist=0.02, new_track_at_beginning_dist=0.005, 
                new_track_middle_dist=0.02, track_disappear_at_end_dist=0.005, track_disappear_at_middle_dist=0.02):
        """Create a new data association object.

        See Florian Pfaff. Multitarget Tracking Using Orientation Estimation for Optical Belt Sorting. Chapter 3.

        Args:
            num_timesteps (int):                    Maximal number of measurements in a track
            rotate_columns (Boolean):               Not in use anymore.
            visualization_path (String):            Path for MTT visualization
            visualize (Boolean):                    Create a visualization of the MTT (time expensive!)
            matching_algorithm (String):            'local' or 'global' -- Matching algorithm
            delta_start_end_phase (double):         Length of start_ and end_phase
            no_change_dist (double):                1/2 Lower right distance entries for "no change"
            new_track_at_beginning_dist (double):   Distance value for additional rows if x-value of measurement < start_phase
            new_track_middle_dist (double):         Distance value for additional rows if x-value of measurement >= start_phase
            track_disappear_at_end_dist (double):   Distance value for additional columns if x-value of measurement > end_phase
            track_disappear_at_middle_dist (double): Distance value for additional columns if x-value of measurement <= end_phase
            """    
        self.num_timesteps = num_timesteps
        self.rotate_columns = rotate_columns

        self.delta_start_end_phase = delta_start_end_phase
        self.no_change_dist = no_change_dist
        self.new_track_at_beginning_dist = new_track_at_beginning_dist
        self.new_track_middle_dist = new_track_middle_dist
        self.track_disappear_at_end_dist = track_disappear_at_end_dist
        self.track_disappear_at_middle_dist = track_disappear_at_middle_dist

        self.matching_algorithm = matching_algorithm
        self.visualization_path = visualization_path
        self.visualize = visualize
        self.current_time_step = 0
        self.max_new_meas = 100

    def associate_data(self, particle_time_list, track_manager, model_manager, belt_limits):
        """Perform the data association algorithm.

        See Florian Pfaff. Multitarget Tracking Using Orientation Estimation for Optical Belt Sorting. Chapter 3.

        Args:
            particle_time_list (list):  List of np arrays. One entry for every timestep. Has multiple particles in each timestep.
                                        Each particle has [id, x, y].
            track_manager:              TrackManager object
            model_manager:              ModelManager object
            belt_limits (np.array):     Limits of the belt [[x_min, x_max],[y_min, y_max]]

        Returns:
            All tracks from track_manager.get_tracks()
            All particle ids
        """
        old_measurements = None
        # Make directory for visualization
        if self.visualize:
            shutil.rmtree(self.visualization_path, ignore_errors=True)
            os.makedirs(self.visualization_path)
        # Set start and end phase for data association weighting
        start_phase = belt_limits[0,0] + self.delta_start_end_phase
        end_phase = belt_limits[0,1] - self.delta_start_end_phase
        # Iterate over timesteps
        all_particle_ids = dict()
        for time_step in range(self.num_timesteps):
            logging.info('step {} / {}'.format(time_step, self.num_timesteps))

            if self.visualize:
                plt.title('Time step: {}'.format(time_step))
                plt.xlim((belt_limits[0,0], belt_limits[0,1]))
                plt.ylim((belt_limits[1,0], belt_limits[1,1]))

            self.current_time_step = time_step
            
            ## Get the measurements and particle ids at the current time step
            particles = particle_time_list[time_step]
            measurements = particles[:,1:]
            particle_ids = particles[:,0]
            for p_id in particle_ids:
                all_particle_ids[int(p_id)]=True
            ## Predict new belt position for each track
            predictions = track_manager.get_predictions(model_manager)
            prediction_ids = list(predictions.keys())
            prediction_values = np.array(list(predictions.values()))
            prediction_is_alive_probabilities = list(map(lambda x: track_manager.get_alive_probability(x), prediction_ids))
            
            n_meas = measurements.shape[0]
            n_pred = prediction_values.shape[0]

            if old_measurements is not None:
                if len(old_measurements) != n_pred:
                    logging.error('number old_measurements different from number predictions!')
                    code.interact(local=dict(globals(), **locals()))
                if self.visualize:
                    for prediction_id in prediction_ids:
                        if old_measurements[prediction_id][1]:
                            plt.scatter([old_measurements[prediction_id][0][0]], [old_measurements[prediction_id][0][1]],
                                        c='cyan', label='old measurement')
                        else:
                            plt.scatter([old_measurements[prediction_id][0][0]], [old_measurements[prediction_id][0][1]],
                                        c='yellow', label='old measurement artificial')
                        start = old_measurements[prediction_id][0]
                        end = predictions[prediction_id]
                        line = np.stack((start, end), axis=0)
                        plt.plot(line[:, 0], line[:, 1], c='purple', label='prediction step')

            if self.visualize:
                if n_meas != 0:
                    plt.scatter(measurements[:, 0], measurements[:, 1], c='blue', label='measurement')
                if n_pred != 0:
                    plt.scatter(prediction_values[:, 0], prediction_values[:, 1], c='red',
                                label='prediction')

            ## Build distance matrix for association
            n_new_rows = n_meas
            n_new_cols = n_pred
            distance_matrix = np.full([n_pred + n_new_rows, n_meas + n_new_cols], np.inf)
            if distance_matrix.size == 0:
                continue
            # Calculate distance matrix
            if n_meas>0 and n_pred>0:
                M = np.swapaxes(np.repeat(np.expand_dims(measurements,1), n_pred, axis=1), 0, 1)
                P = np.repeat(np.expand_dims(prediction_values, 1), n_meas, axis=1)
                distance_matrix[:n_pred, :n_meas] = np.linalg.norm(M-P, axis=2)
            # Add additional rows for measurements without tracks 
            if n_new_rows>0:
                dist = self.new_track_middle_dist + (self.new_track_at_beginning_dist - self.new_track_middle_dist) * (measurements[:,0] < start_phase)
                new_rows = np.transpose(np.repeat(np.expand_dims(dist, 1), n_new_rows, 1))
                distance_matrix[n_pred:, 0:n_meas] = new_rows
            # and additional columns for tracks without measurement
            if n_new_cols>0:
                dist = self.track_disappear_at_middle_dist + (self.track_disappear_at_end_dist - self.track_disappear_at_middle_dist) * (prediction_values[:,0] > end_phase)
                new_cols = np.repeat(np.expand_dims(dist, 1), n_new_cols, 1)
                distance_matrix[0:n_pred, n_meas:] = new_cols
            # Add additional matrix for no change
            if n_new_rows>0 and n_new_cols>0:
                no_change_mat = 2*self.no_change_dist * np.ones([n_new_rows, n_new_cols])
                distance_matrix[n_pred:, n_meas:] = no_change_mat
            ## Associate predictions and measurements based on the distance matrix
            if self.matching_algorithm == 'local':
               prediction_idxs, measurement_idxs = nearest_neighbour(distance_matrix)
            elif self.matching_algorithm == 'global':
               prediction_idxs, measurement_idxs = linear_sum_assignment(distance_matrix)
        
            ## Create new, update existing and delete expired tracks
            counts = np.zeros([4], dtype=np.int32)
            old_measurements = {}
            for idx in range(len(measurement_idxs)):
                if measurement_idxs[idx] < n_meas and prediction_idxs[idx] < n_pred:
                    # The measurement was associated to an existing track
                    counts[0] += 1
                    prediction_id = prediction_ids[prediction_idxs[idx]]
                    particle_id = particle_ids[measurement_idxs[idx]]
                    #
                    measurement = measurements[measurement_idxs[idx]]
                    track_manager.real_track_real_measurement(prediction_id, particle_id, measurement, model_manager)
                    old_measurements[prediction_id] = (measurement, True)
                    #
                    if self.visualize: 
                        line = np.stack((measurements[measurement_idxs[idx]], prediction_values[prediction_idxs[idx]]),
                                    axis=0)
                        plt.plot(line[:, 0], line[:, 1], c='green')
                #
                elif measurement_idxs[idx] >= n_meas and prediction_idxs[idx] < n_pred:
                    # No measurement associated to existing track
                    counts[1] += 1
                    # feed it back its own prediction as measurement
                    prediction_id = prediction_ids[prediction_idxs[idx]]
                    prediction = prediction_values[prediction_idxs[idx]]
                    is_still_alive = track_manager.real_track_pseudo_measurement(prediction_id, prediction, model_manager)
                    if is_still_alive:
                        old_measurements[prediction_id] = (prediction, False)
                    else:
                        logging.debug('track finished!')
                        if self.visualize: plt.scatter([prediction[0]], [prediction[1]], c='black', label='track end')
                    #
                    if self.visualize: 
                        circle = plt.Circle(prediction, self.no_change_dist, color='blue', fill=False, label='artificial track')
                        plt.gcf().gca().add_artist(circle)
                        # only for visualization purposes
                        pseudo_measurement = np.array(
                            [prediction[0] + self.no_change_dist, prediction[1]])
                        line = np.stack((pseudo_measurement, prediction), axis=0)
                        plt.plot(line[:, 0], line[:, 1], c='green')
                #
                elif measurement_idxs[idx] < n_meas and prediction_idxs[idx] >= n_pred:
                    # Measurement associated to new track
                    counts[2] += 1
                    #
                    measurement = measurements[measurement_idxs[idx]]
                    particle_id = particle_ids[measurement_idxs[idx]]
                    prediction_id = track_manager.pseudo_track_real_measurement(measurement, particle_id, time_step, model_manager)
                    old_measurements[prediction_id] = (measurement, True)
                    #
                    if self.visualize:
                        circle = plt.Circle(measurement,
                                            self.no_change_dist,
                                            color='red',
                                            fill=False,
                                            label='artificial track')
                        plt.gcf().gca().add_artist(circle)
                        pseudo_prediction = np.array(
                            [measurement[0] + self.no_change_dist, measurement[1]])
                        line = np.stack((measurement, pseudo_prediction), axis=0)
                        plt.plot(line[:, 0], line[:, 1], c='green', label='matching')
                else:
                    counts[3] += 1
            logging.debug(list(counts))

            if self.visualize:
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc="upper left", ncol=2)
                plt.savefig(os.path.join(self.visualization_path, '{:05d}'.format(time_step)))
                plt.clf()
        #
        return track_manager.get_tracks(), all_particle_ids
