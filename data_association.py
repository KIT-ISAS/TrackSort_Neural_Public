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


# definition of my own nearest neighbour method
def nearest_neighbour(weight_matrix):
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
    logging.error('something went wrong in nearest_neighbour!')
    code.interact(local=dict(globals(), **locals()))


class DataAssociation(object):
    def __init__(self, num_timesteps, rotate_columns, visualization_path, visualize,
                distance_threshold, positional_probabilities, is_alive_probability_weighting, matching_algorithm):
        self.num_timesteps = num_timesteps
        self.rotate_columns = rotate_columns
        self.distance_threshold = distance_threshold
        self.positional_probabilities = positional_probabilities
        self.is_alive_probability_weighting = is_alive_probability_weighting
        self.matching_algorithm = matching_algorithm
        self.visualization_path = visualization_path
        self.visualize = visualize
        self.current_time_step = 0

    def associate_data(self, data_source, track_manager, model_manager):
        old_measurements = None
        shutil.rmtree(self.visualization_path, ignore_errors=True)
        os.makedirs(self.visualization_path)
        for time_step in range(self.num_timesteps):
            logging.info('step {} / {}'.format(time_step, self.num_timesteps))

            if self.visualize:
                plt.title('Time step: {}'.format(time_step))
                if not self.rotate_columns:
                    plt.xlim((-0.1, 1.3))  # TODO more sophisticated solution to this problem???
                    plt.ylim((-0.1, 1.5))

                if self.rotate_columns:
                    plt.xlim((0.3, 0.8))
                    plt.ylim((0.0, 0.2))

            self.current_time_step = time_step
            
            ## Get the measurements at the current time step
            measurements = data_source.get_measurement_at_timestep_list(time_step)
            
            ## Predict new belt position for each track
            predictions = track_manager.get_predictions(model_manager)
            prediction_ids = list(predictions.keys())
            prediction_values = list(predictions.values())
            prediction_is_alive_probabilities = list(map(lambda x: track_manager.get_alive_probability(x), prediction_ids))
            
            #
            if old_measurements is not None:
                if len(old_measurements) != len(prediction_values):
                    logging.error('number old_measurements different from number predictions!')
                    code.interact(local=dict(globals(), **locals()))
                for idx, prediction_id in enumerate(prediction_ids):
                    if old_measurements[prediction_id][1]:
                        if self.visualize: plt.scatter([old_measurements[prediction_id][0][0]], [old_measurements[prediction_id][0][1]],
                                    c='cyan', label='old measurement')
                    else:
                        if self.visualize: plt.scatter([old_measurements[prediction_id][0][0]], [old_measurements[prediction_id][0][1]],
                                    c='yellow', label='old measurement artificial')
                    start = old_measurements[prediction_id][0]
                    end = predictions[prediction_id]
                    line = np.stack((start, end), axis=0)
                    if self.visualize: plt.plot(line[:, 0], line[:, 1], c='purple', label='prediction step')

            if len(measurements) != 0:
                if self.visualize: plt.scatter(np.array(measurements)[:, 0], np.array(measurements)[:, 1], c='blue', label='measurement')
            if prediction_values != []:
                if self.visualize: plt.scatter(np.array(prediction_values)[:, 0], np.array(prediction_values)[:, 1], c='red',
                            label='prediction')

            ## Build distance matrix for association
            # why isn't infinity working anymore???
            distance_matrix = 10000 * np.ones([2 * len(measurements) + len(prediction_values), 2 * len(prediction_values) + len(measurements)])
            if distance_matrix.size == 0:
                continue
            #
            for measurement_nr in range(len(measurements)):
                for prediction_nr in range(len(prediction_values)):
                    distance_matrix[measurement_nr][prediction_nr] = np.linalg.norm(
                        measurements[measurement_nr] - prediction_values[prediction_nr])
            #
            for measurement_nr in range(len(measurements)):
                distance_matrix[measurement_nr][len(prediction_values) + measurement_nr] = self.distance_threshold \
                    * math.pow(1.0 + (1.0 - measurements[measurement_nr][0]), self.positional_probabilities)
                distance_matrix[len(measurements) + len(prediction_values) + measurement_nr][
                    len(prediction_values) + measurement_nr] = 1.1 * self.distance_threshold\
                    * math.pow(1.0 + (1.0 - measurements[measurement_nr][0]), self.positional_probabilities)
            #
            for prediction_nr, _ in enumerate(prediction_values):
                # code.interact(local=dict(globals(), **locals()))
                distance_matrix[len(measurements) + prediction_nr][prediction_nr] = self.distance_threshold \
                    * math.pow(1.0 + prediction_is_alive_probabilities[prediction_nr], self.is_alive_probability_weighting) \
                    * math.pow(1.0 + prediction_values[prediction_nr][0], self.positional_probabilities)
                distance_matrix[len(measurements) + prediction_nr][
                    len(measurements) + len(prediction_values) + prediction_nr] = 1.1 * self.distance_threshold \
                    * math.pow(1.0 + prediction_is_alive_probabilities[prediction_nr], self.is_alive_probability_weighting) \
                    * math.pow(1.0 + prediction_values[prediction_nr][0], self.positional_probabilities)
            #
            if self.matching_algorithm == 'global': 
                distance_matrix[-len(measurements):,-len(prediction_values)] = 1.2 * self.distance_threshold \
                        * math.pow(2.0, self.is_alive_probability_weighting)
                if len(measurements) >= len(prediction_values):
                    distance_matrix_a = 10000 * np.ones([len(measurements), len(prediction_values) + len(measurements)])
                    distance_matrix_b = 1.2 * self.distance_threshold * np.ones([len(measurements), len(prediction_values)]) \
                        * math.pow(2.0, self.is_alive_probability_weighting)
                    distance_matrix_c = np.concatenate([distance_matrix_a, distance_matrix_b], axis=1)
                    distance_matrix = np.concatenate([distance_matrix, distance_matrix_c], axis=0)
                else:
                    distance_matrix_a = 10000 * np.ones([len(prediction_values) + len(measurements), len(prediction_values)])
                    distance_matrix_b = 1.2 * self.distance_threshold * np.ones([len(measurements), len(prediction_values)]) \
                        * math.pow(2.0, self.is_alive_probability_weighting)
                    distance_matrix_c = np.concatenate([distance_matrix_a, distance_matrix_b], axis=0)
                    distance_matrix = np.concatenate([distance_matrix, distance_matrix_c], axis=1)
            
            ## Associate predictions and measurements based on the distance matrix
            if self.matching_algorithm == 'local':
                measurement_idxs, prediction_idxs = nearest_neighbour(distance_matrix)
            elif self.matching_algorithm == 'global':
                measurement_idxs, prediction_idxs = linear_sum_assignment(distance_matrix)
            
            ## Create new, update existing and delete expired tracks
            counts = np.zeros([4], dtype=np.int32)
            old_measurements = {}
            for idx in range(len(measurement_idxs)):
                if measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] < len(prediction_values):
                    # The measurement was associated to an existing track
                    counts[0] += 1
                    prediction_id = prediction_ids[prediction_idxs[idx]]
                    #
                    track_manager.real_track_real_measurement(prediction_id, measurements[measurement_idxs[idx]], model_manager)
                    old_measurements[prediction_id] = (measurements[measurement_idxs[idx]], True)
                    #
                    line = np.stack((measurements[measurement_idxs[idx]], prediction_values[prediction_idxs[idx]]),
                                    axis=0)
                    if self.visualize: plt.plot(line[:, 0], line[:, 1], c='green')
                #
                elif measurement_idxs[idx] >= len(measurements) and prediction_idxs[idx] < len(prediction_values):
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
                    if self.visualize: circle = plt.Circle(prediction, self.distance_threshold, color='blue', fill=False, label='artificial track')
                    if self.visualize: plt.gcf().gca().add_artist(circle)
                    # only for visualization purposes
                    pseudo_measurement = np.array(
                        [prediction[0] + self.distance_threshold, prediction[1]])
                    line = np.stack((pseudo_measurement, prediction), axis=0)
                    if self.visualize: plt.plot(line[:, 0], line[:, 1], c='green')
                #
                elif measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] >= len(prediction_values):
                    # Measurement associated to new track
                    counts[2] += 1
                    #
                    measurement = measurements[measurement_idxs[idx]]
                    prediction_id = track_manager.pseudo_track_real_measurement(measurement, time_step, model_manager)
                    old_measurements[prediction_id] = (measurement, True)
                    #
                    if self.visualize:
                        circle = plt.Circle(measurement,
                                            self.distance_threshold,
                                            color='red',
                                            fill=False,
                                            label='artificial track')
                        plt.gcf().gca().add_artist(circle)
                    #
                    pseudo_prediction = np.array(
                        [measurement[0] + self.distance_threshold, measurement[1]])
                    line = np.stack((measurement, pseudo_prediction), axis=0)
                    if self.visualize: plt.plot(line[:, 0], line[:, 1], c='green', label='matching')
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
        return track_manager.tracks
