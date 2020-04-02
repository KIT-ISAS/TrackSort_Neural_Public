import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import code  # code.interact(local=dict(globals(), **locals()))
import shutil
import math
import os
import logging

from matplotlib.patches import Ellipse

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
    def __init__(self, global_config):
        self.global_config = global_config
        if self.global_config['dataset_type'] == 'FakeDataset':
            self.data_source = FakeDataSet(global_config=global_config)
        elif self.global_config['dataset_type'] == 'CsvDataset':
            self.data_source = CsvDataSet(global_config=global_config, **global_config['CsvDataSet'])
            self.global_config['num_timesteps'] = self.data_source.get_num_timesteps()
        else:
            logging.debug(str(self.global_config['dataset_type'] + ' is no valid data source!'))
            raise ValueError("dataset_type invalid")
        self.track_manager = TrackManager(global_config, self.data_source)

    def associate_data(self):
        """
        Main loop for the data association


        """
        # visualization paths
        shutil.rmtree(self.global_config['visualization_path'], ignore_errors=True)
        os.makedirs(self.global_config['visualization_path'])

        old_measurements = None

        for time_step in range(self.global_config['num_timesteps']):
            logging.info('step {} / {}'.format(time_step, self.global_config['num_timesteps']))

            if self.global_config['visualize']:
                plt.title('Time step: {}'.format(time_step))
                if not self.global_config['CsvDataSet']['rotate_columns']:
                    plt.xlim((-0.1, 1.3))  # TODO more sophisticated solution to this problem???
                    plt.ylim((-0.1, 1.5))

                if self.global_config['CsvDataSet']['rotate_columns']:
                    plt.xlim((0.3, 0.8))
                    plt.ylim((0.0, 0.2))

            self.global_config['current_time_step'] = time_step

            measurements = self.data_source.get_measurement_at_timestep_list(time_step)
            predictions, variances = self.track_manager.get_predictions()

            prediction_ids = list(predictions.keys())
            prediction_values = list(predictions.values())
            prediction_is_alive_probabilities = list(
                map(lambda x: self.track_manager.get_alive_probability(x), prediction_ids))

            # visualize:
            if self.global_config['visualize']:

                # old measurements, old artificial measurements
                if old_measurements is not None:
                    if len(old_measurements) != len(prediction_values):
                        logging.error('number old_measurements different from number predictions!')
                        code.interact(local=dict(globals(), **locals()))

                    for idx, prediction_id in enumerate(prediction_ids):
                        if old_measurements[prediction_id][1]:
                            plt.scatter([old_measurements[prediction_id][0][0]],
                                        [old_measurements[prediction_id][0][1]],
                                        c='cyan', label='old measurement')
                        else:
                            plt.scatter([old_measurements[prediction_id][0][0]],
                                        [old_measurements[prediction_id][0][1]],
                                        c='yellow', label='old measurement artificial')

                        # visualize: prediction step
                        start = old_measurements[prediction_id][0]
                        end = predictions[prediction_id]
                        line = np.stack((start, end), axis=0)
                        plt.plot(line[:, 0], line[:, 1], c='purple', label='prediction step')

                # visualize: new measurements
                if len(measurements) != 0:
                    plt.scatter(np.array(measurements)[:, 0], np.array(measurements)[:, 1], c='blue',
                                label='measurement')

                # visualize: predictions
                if len(prediction_values) > 0:
                    plt.scatter(np.array(prediction_values)[:, 0], np.array(prediction_values)[:, 1], c='red',
                                label='prediction')

                    # predicted uncertainty as ellipse
                    if variances is not None:
                        for prediction_id, prediction in predictions.items():
                            variance = variances[prediction_id]
                            stddev = np.sqrt(variance)
                            circle = Ellipse(prediction, stddev[0] * 2, stddev[1] * 2, color='grey',
                                             fill=True, alpha=0.1, label='prediction uncertainty')
                            plt.gcf().gca().add_artist(circle)

            # pseudo infinity value
            infinity = 10000

            # construct distance matrix for (global/local) nearest neighbour
            # ... initialize with infinite distance
            # structure of the matrix:
            #    A | B | C
            #    ---------
            #    D | E | F
            #    ---------
            #    G | H | I

            distance_matrix = infinity * np.ones([2 * len(measurements) + len(prediction_values),
                                                  2 * len(prediction_values) + len(measurements)])

            if distance_matrix.size == 0:
                continue

            # constants
            distance_threshold = self.global_config['distance_threshold']
            positional_probabilities = self.global_config['positional_probabilities']
            is_alive_probability_weighting = self.global_config['is_alive_probability_weighting']

            # Block matrix: A
            #  ... L2 distance between predictions and measurements
            for prediction_nr, prediction in enumerate(prediction_values):
                for measurement_nr, measurement in enumerate(measurements):
                    if variances is not None:
                        # standardized Euclidean distance
                        distance_matrix[measurement_nr][prediction_nr] = \
                            np.linalg.norm((measurement - prediction)/variances[prediction_ids[prediction_nr]])
                    else:
                        # euclidean distance (=L2 norm)
                        distance_matrix[measurement_nr][prediction_nr] = np.linalg.norm(measurement - prediction)

            # Block matrix: B
            # ... measurements to their artificial predictions
            for measurement_nr, measurement in enumerate(measurements):
                distance_matrix[measurement_nr][len(prediction_values) + measurement_nr] = \
                    distance_threshold * math.pow(1.0 + (1.0 - measurement[0]), positional_probabilities)

            # Block matrix: H
            for measurement_nr, measurement in enumerate(measurements):
                distance_matrix[len(measurements) + len(prediction_values) + measurement_nr][
                    len(prediction_values) + measurement_nr] = 1.1 * distance_threshold \
                                                               * math.pow(1.0 + (1.0 - measurement[0]),
                                                                          positional_probabilities)

            # Block matrix: D
            for prediction_nr, prediction in enumerate(prediction_values):
                distance_matrix[len(measurements) + prediction_nr][prediction_nr] = distance_threshold \
                                                                                    * math.pow(
                    1.0 + prediction_is_alive_probabilities[prediction_nr],
                    is_alive_probability_weighting) \
                                                                                    * math.pow(
                    1.0 + prediction_values[prediction_nr][0], positional_probabilities)

            # Block matrix: F
            for prediction_nr, prediction in enumerate(prediction_values):
                distance_matrix[len(measurements) + prediction_nr][
                    len(prediction_values) + len(measurements) + prediction_nr] = 1.1 * distance_threshold \
                                                                                  * math.pow(
                    1.0 + prediction_is_alive_probabilities[prediction_nr],
                    is_alive_probability_weighting) \
                                                                                  * math.pow(
                    1.0 + prediction_values[prediction_nr][0], positional_probabilities)

            #
            if self.global_config['matching_algorithm'] == 'global':
                distance_matrix[-len(measurements):, -len(prediction_values)] = 1.2 * distance_threshold \
                                                                                * math.pow(2.0, is_alive_probability_weighting)

                if len(measurements) >= len(prediction_values):
                    distance_matrix_a = infinity * np.ones([len(measurements), len(prediction_values) + len(measurements)])
                    distance_matrix_b = 1.2 * distance_threshold * np.ones(
                        [len(measurements), len(prediction_values)]) \
                                        * math.pow(2.0, is_alive_probability_weighting)
                    distance_matrix_c = np.concatenate([distance_matrix_a, distance_matrix_b], axis=1)
                    distance_matrix = np.concatenate([distance_matrix, distance_matrix_c], axis=0)
                else:
                    distance_matrix_a = infinity * np.ones(
                        [len(prediction_values) + len(measurements), len(prediction_values)])
                    distance_matrix_b = 1.2 * distance_threshold * np.ones(
                        [len(measurements), len(prediction_values)]) \
                                        * math.pow(2.0, is_alive_probability_weighting)
                    distance_matrix_c = np.concatenate([distance_matrix_a, distance_matrix_b], axis=0)
                    distance_matrix = np.concatenate([distance_matrix, distance_matrix_c], axis=1)
            #

            if self.global_config['matching_algorithm'] == 'local':
                measurement_idxs, prediction_idxs = nearest_neighbour(distance_matrix)
            elif self.global_config['matching_algorithm'] == 'global':
                measurement_idxs, prediction_idxs = linear_sum_assignment(distance_matrix)
            #
            counts = np.zeros([4], dtype=np.int32)
            old_measurements = {}

            for idx in range(len(measurement_idxs)):

                # A: real measurement <-> real prediction
                if measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] < len(prediction_values):
                    counts[0] += 1
                    prediction_id = prediction_ids[prediction_idxs[idx]]
                    #
                    self.track_manager.real_track_real_measurement(prediction_id, measurements[measurement_idxs[idx]])
                    old_measurements[prediction_id] = (measurements[measurement_idxs[idx]], True)
                    #
                    line = np.stack((measurements[measurement_idxs[idx]], prediction_values[prediction_idxs[idx]]),
                                    axis=0)
                    if self.global_config['visualize']:
                        plt.plot(line[:, 0], line[:, 1], c='green')

                # D: artificial measurement <-> real prediction
                elif measurement_idxs[idx] >= len(measurements) and prediction_idxs[idx] < len(prediction_values):
                    counts[1] += 1
                    # feed it back its own prediction as measurement
                    prediction_id = prediction_ids[prediction_idxs[idx]]
                    prediction = prediction_values[prediction_idxs[idx]]
                    is_still_alive = self.track_manager.real_track_pseudo_measurement(prediction_id, prediction)
                    if is_still_alive:
                        old_measurements[prediction_id] = (prediction, False)
                    else:
                        logging.debug('track finished!')
                        if self.global_config['visualize']:
                            plt.scatter([prediction[0]], [prediction[1]], c='black', label='track end')
                    #
                    if self.global_config['visualize']:
                        circle = Ellipse(prediction, distance_threshold*2, distance_threshold*2, color='blue', fill=False, label='artificial track')
                        plt.gcf().gca().add_artist(circle)

                    # only for visualization purposes
                    pseudo_measurement = np.array(
                        [prediction[0] + distance_threshold, prediction[1]])
                    line = np.stack((pseudo_measurement, prediction), axis=0)
                    if self.global_config['visualize']:
                        plt.plot(line[:, 0], line[:, 1], c='green')

                # B: real measurement <-> artificial prediction
                elif measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] >= len(prediction_values):
                    counts[2] += 1
                    #
                    measurement = measurements[measurement_idxs[idx]]
                    prediction_id = self.track_manager.pseudo_track_real_measurement(measurement, time_step)
                    old_measurements[prediction_id] = (measurement, True)
                    #
                    if self.global_config['visualize']:
                        circle = Ellipse(measurement, distance_threshold*2, distance_threshold*2, color='red', fill=False, label='artificial track')
                        plt.gcf().gca().add_artist(circle)
                    #
                    pseudo_prediction = np.array(
                        [measurement[0] + distance_threshold, measurement[1]])
                    line = np.stack((measurement, pseudo_prediction), axis=0)
                    if self.global_config['visualize']:
                        plt.plot(line[:, 0], line[:, 1], c='green', label='matching')
                else:
                    counts[3] += 1
            logging.debug(list(counts))

            if self.global_config['visualize']:
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc="upper left", ncol=2)
                plt.savefig(os.path.join(self.global_config['visualization_path'], '{:05d}'.format(time_step)))
                plt.clf()
        #
        return self.track_manager.tracks
