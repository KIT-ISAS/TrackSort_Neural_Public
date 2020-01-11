import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import code  # code.interact(local=dict(globals(), **locals()))
import copy, shutil, os

from track_manager import TrackManager
from data_manager import FakeDataSet, CsvDataSet


class DataAssociation(object):
    def __init__(self, global_config):
        self.global_config = global_config
        if self.global_config['dataset_type'] == 'FakeDataset':
            self.data_source = FakeDataSet(global_config=global_config)
        else:
            self.data_source = CsvDataSet(global_config=global_config)
        self.track_manager = TrackManager(global_config, self.data_source)

    def associate_data(self):
        old_measurements = None
        shutil.rmtree(self.global_config['visualization_path'], ignore_errors=True)
        os.makedirs(self.global_config['visualization_path'])
        for time_step in range(self.global_config['num_timesteps']):
            print('')
            print('step ' + str(time_step) + ' / ' + str(self.global_config['num_timesteps']))
            plt.title('Time step: {}'.format(time_step))
            plt.xlim((-0.1, 1.3))
            plt.ylim((-0.1, 1.1))
            self.global_config['current_time_step'] = time_step
            #
            measurements = self.data_source.get_measurement_at_timestep_list(time_step)
            #
            predictions = self.track_manager.get_predictions()
            prediction_ids = list(predictions.keys())
            prediction_values = list(predictions.values())
            #
            if old_measurements is not None:
                if len(old_measurements) != len(prediction_values):
                    print('number old_measurements different from number predictions!')
                    code.interact(local=dict(globals(), **locals()))
                for idx, prediction_id in enumerate(prediction_ids):
                    if old_measurements[prediction_id][1]:
                        plt.scatter([old_measurements[prediction_id][0][0]], [old_measurements[prediction_id][0][1]],
                                    c='cyan')  # , label='old measurement')
                    else:
                        plt.scatter([old_measurements[prediction_id][0][0]], [old_measurements[prediction_id][0][1]],
                                    c='yellow')  # , label='old measurement artificial')
                    start = old_measurements[prediction_id][0]
                    end = predictions[prediction_id]
                    line = np.stack((start, end), axis=0)
                    plt.plot(line[:, 0], line[:, 1], c='purple')
            # print('in associate_data')
            # code.interact(local=dict(globals(), **locals()))
            if len(measurements) != 0:
                plt.scatter(np.array(measurements)[:, 0], np.array(measurements)[:, 1], c='blue', label='measurement')
            if prediction_values != []:
                plt.scatter(np.array(prediction_values)[:, 0], np.array(prediction_values)[:, 1], c='red',
                            label='prediction')

            # why isn't infinity working anymore???
            distance_matrix = 10000 * np.ones(
                [2 * len(measurements) + len(prediction_values), 2 * len(prediction_values) + len(measurements)])
            #
            for measurement_nr in range(len(measurements)):
                for prediction_nr in range(len(prediction_values)):
                    distance_matrix[measurement_nr][prediction_nr] = np.linalg.norm(
                        measurements[measurement_nr] - prediction_values[prediction_nr])
            #
            for measurement_nr in range(len(measurements)):
                distance_matrix[measurement_nr][len(prediction_values) + measurement_nr] = self.global_config[
                    'distance_threshhold']
                distance_matrix[len(measurements) + len(prediction_values) + measurement_nr][
                    len(prediction_values) + measurement_nr] = 1.1 * self.global_config['distance_threshhold']
            #
            for prediction_nr in range(len(prediction_values)):
                distance_matrix[len(measurements) + prediction_nr][prediction_nr] = self.global_config[
                    'distance_threshhold']
                distance_matrix[len(measurements) + prediction_nr][
                    len(measurements) + len(prediction_values) + prediction_nr] = 1.1 * self.global_config[
                    'distance_threshhold']
            #
            #print('before matching')
            #code.interact(local=dict(globals(), **locals()))
            measurement_idxs, prediction_idxs = linear_sum_assignment(distance_matrix)
            #
            counts = np.zeros([4], dtype=np.int32)
            old_measurements = {}
            for idx in range(len(measurement_idxs)):
                if measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] < len(prediction_values):
                    # print('realreal')
                    counts[0] += 1
                    prediction_id = prediction_ids[prediction_idxs[idx]]
                    #
                    self.track_manager.real_track_real_measurement(prediction_id, measurements[measurement_idxs[idx]])
                    old_measurements[prediction_id] = (measurements[measurement_idxs[idx]], True)
                    #
                    line = np.stack((measurements[measurement_idxs[idx]], prediction_values[prediction_idxs[idx]]),
                                    axis=0)
                    plt.plot(line[:, 0], line[:, 1], c='green')
                #
                elif measurement_idxs[idx] >= len(measurements) and prediction_idxs[idx] < len(prediction_values):
                    # print('realpseudo')
                    counts[1] += 1
                    # feed it back its own prediction as measurement
                    prediction_id = prediction_ids[prediction_idxs[idx]]
                    prediction = prediction_values[prediction_idxs[idx]]
                    is_still_alive = self.track_manager.real_track_pseudo_measurement(prediction_id, prediction)
                    if is_still_alive:
                        old_measurements[prediction_id] = (prediction, False)
                    else:
                        print('track finished!')
                        plt.scatter([prediction[0]], [prediction[1]], c='black')
                    #
                    circle = plt.Circle(prediction, self.global_config['distance_threshhold'], color='blue', fill=False)
                    plt.gcf().gca().add_artist(circle)
                    # only for visualization purposes
                    pseudo_measurement = np.array(
                        [prediction[0] + self.global_config['distance_threshhold'], prediction[1]])
                    line = np.stack((pseudo_measurement, prediction), axis=0)
                    plt.plot(line[:, 0], line[:, 1], c='green')
                #
                elif measurement_idxs[idx] < len(measurements) and prediction_idxs[idx] >= len(prediction_values):
                    # print('pseudoreal')
                    counts[2] += 1
                    #
                    measurement = measurements[measurement_idxs[idx]]
                    prediction_id = self.track_manager.pseudo_track_real_measurement(measurement, time_step)
                    old_measurements[prediction_id] = (measurement, True)
                    #
                    circle = plt.Circle(measurement, self.global_config['distance_threshhold'], color='red', fill=False)
                    plt.gcf().gca().add_artist(circle)
                    #
                    pseudo_prediction = np.array(
                        [measurement[0] + self.global_config['distance_threshhold'], measurement[1]])
                    line = np.stack((measurement, pseudo_prediction), axis=0)
                    plt.plot(line[:, 0], line[:, 1], c='green')
                else:
                    # print('pseudopseudo')
                    counts[3] += 1
            print(list(counts))
            plt.legend(loc="upper left")
            plt.savefig(self.global_config['visualization_path'] + '{:05d}'.format(time_step))
            plt.clf()
        # print('in associate_data: ' + str(time_step))
        # code.interact(local=dict(globals(), **locals()))
        #
        return self.track_manager.tracks
