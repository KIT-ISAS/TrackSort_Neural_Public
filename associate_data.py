import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
from tensorflow.keras import backend as K
import code
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import os, sys, shutil
import random
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import backend as K
from datetime import datetime
from collections import defaultdict

# custom modules
import data
import model
from data import CsvDataSet, FakeDataSet
from model import ModelManager, rnn_model_factory

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
	print('something went wrong in nearest_neighbour!')
	code.interact(local=dict(globals(), **locals()))

num_time_steps = 100
nan_value = 0
batch_size = 64
belt_width = 2000

dataset = data.FakeDataSet(timesteps=num_time_steps, batch_size=batch_size, number_trajectories=130, 
                           additive_noise_stddev=2, additive_target_stddev=20, belt_width=belt_width,
                          nan_value=nan_value)

dataset_train, dataset_test = dataset.get_tf_data_sets_seq2seq_data(normalized=True)

longest_track_time_steps = dataset.longest_track

# dataset.plot_random_tracks(n=15)

LOAD_MODEL = True
# define model
if LOAD_MODEL:
	rnn_model = tf.keras.models.load_model('models/rnn_model_fake_data.h5')
else:
	rnn_model, hash_ = model.rnn_model_factory(
			num_units_first_rnn=16, 
			num_units_second_rnn=16,
			num_units_first_dense=0,
			rnn_model_name='lstm',
			num_time_steps=longest_track_time_steps, 
			batch_size=batch_size, 
			nan_value=nan_value, 
	    unroll=True,
			input_dim=2)
	print(rnn_model.summary())

	optimizer = tf.keras.optimizers.Adam()
	train_step_fn = model.train_step_generator(rnn_model, optimizer)

	total_num_epochs = 1000


	# Train model
	for epoch in range(total_num_epochs):
		# learning rate decay after 100 epochs
		if (epoch+1) % 150 == 0:
			old_lr = K.get_value(optimizer.lr)
			new_lr = old_lr * 0.1
			print("Reducing learning rate from {} to {}.".format(old_lr, new_lr))
			K.set_value(optimizer.lr, new_lr)

		for (batch_n, (inp, target)) in enumerate(dataset_train):
			_ = rnn_model.reset_states()
			loss = train_step_fn(inp, target)	

		print("{}/{}: \t loss={}".format(epoch, total_num_epochs, loss))



shutil.rmtree('visualizations/matching_visualization_local', ignore_errors=True)
os.makedirs('visualizations/matching_visualization_local')

number_tracks = dataset.track_data.shape[0]

# create the model manager
model_manager = model.ModelManager(number_tracks, batch_size, rnn_model)

TIME_STEPS = num_time_steps
MAX_NUM_TRAJECTORIES = number_tracks
SIZE_X = 1.
SIZE_Y = 1.
# the region, where the track is considered to appear / disappear
X_DETECTION_TOLERANCE = SIZE_X / 10.
X_DISAPPEAR_TOLERANCE = 0

# the distance of the pseudo measurements and tracks to the actual belt
X_THRESHOLD = 5./2000.

# generate increasing ids
def id_generator():
    n = 0
    while True:
        yield n
        n += 1

id_seed = id_generator()

all_ids = []

# mm: model_manager
# id => mm_id
id_2_mm = {}

# mm_id => id
mm_2_id = {}

# id => list(observations)
track_history = defaultdict(list)

measurements = None

active_ids = set()

for time_step in range(num_time_steps):
    print('Time step: {}'.format(time_step))
    plt.title('Time step: {}'.format(time_step))
    plt.xlim((0.0,1.0))
    plt.ylim((0.0,1.0))
    predictions_mm_ids = [id_2_mm[i] for i in sorted(active_ids)]
    predictions_ids = [i for i in sorted(active_ids)]
    predictions = model_manager.predict()
    
    predictions_mask = np.zeros(predictions.shape[0], dtype=np.bool)
    predictions_mask[predictions_mm_ids] = True
    
    predictions = predictions[predictions_mask]

    # old measurement
    old_observations = np.array(model_manager.batch_measurements).reshape([model_manager.n_batches * model_manager.batch_size, model_manager.num_dims])[predictions_mask]

    if measurements is not None:
      plt.scatter(measurements[:, 0], measurements[:, 1], c='cyan', label='old measurement')

      for pred_i in range(predictions.shape[0]):
        start = old_observations[pred_i]
        end = predictions[pred_i]
        line = np.stack((start, end), axis=0)
        plt.plot(line[:, 0], line[:, 1], c='red')
    
    
    measurements = dataset.get_measurement_at_timestep(time_step)
    mask = np.any(measurements != [nan_value, nan_value], axis=-1)
    measurements = measurements[mask]
    
    plt.scatter(measurements[:, 0], measurements[:, 1], c='blue', label='measurement')
    plt.scatter(predictions[:, 0], predictions[:, 1], marker='4', c='red', label='prediction')
    # plt.show()
    
    if measurements.shape[0] == 0:
        continue
    
    # 1. particle enters perception: add an artificial track for every measurement at the first 1/10 of the belt
    #   -> for every of these tracks: create an artificial measurement
    mask_new_tracks = measurements[:, 0] < X_DETECTION_TOLERANCE
    count_new_tracks = np.sum(mask_new_tracks)
    
    artificial_predictions = np.stack((np.ones(count_new_tracks)*-X_THRESHOLD,
                                        measurements[mask_new_tracks][:, 1]), axis=-1)
    
    # 2. particle leaves perception: the predicted particles which are in the terminal region of the belt
    mask_end_measurements = predictions[:, 0] > SIZE_X - X_DISAPPEAR_TOLERANCE
    count_end_measurements = np.sum(mask_end_measurements)
    artificial_measurements = np.stack((np.ones(count_end_measurements)*(SIZE_X + X_THRESHOLD),
                                        predictions[mask_end_measurements][:, 1]), axis=-1)
    
    # Distance matrix
    all_measurements = np.concatenate((measurements, artificial_measurements))
    all_predictions = np.concatenate((predictions, artificial_predictions))
    distances = distance_matrix(all_measurements, all_predictions) ** 2

    # ToDo: set the special distances for the artificial components
    #   (the artificial measurements and predictions are close together)
    # 1. artificial measurements (where tracks end) are only connected to the
    #    provocing track and one 
    large_value = 5
    pseudo_distance = 0.1

    distances[measurements.shape[0]:, :] = large_value
    for measurement_idx, prediction_idx in enumerate(np.where(mask_end_measurements)[0]):
      # the distance between the pseudo measurement and the track
      distance_value = np.abs(predictions[prediction_idx, 0] - (SIZE_X + X_THRESHOLD))
      print('dist', distance_value)
      distances[measurements.shape[0]+measurement_idx, prediction_idx] = distance_value

      # What if the track doesn't match the pseudo measurement? (for example: because there is a better measurement for the track)
      #  -> then we want the pseudo measurement to match a pseudo track
      #      -> therefore, we create a new pseudo track with the following distances:
      #             - to real measurements: infinty
      #             - to pseudo measurements: 0.1 (pseudo_distance)
      #                (Attention. 0.1 is an arbitrary number which should actually be a sigma value!!!)
      new_column = np.ones(distances.shape[0])[:, np.newaxis] * large_value
      new_column[measurements.shape[0]:] = pseudo_distance
      distances = np.hstack((distances, new_column))

      # Now that we created a new column, we must create a new placeholder
      #   row. For the column to match.
      new_row = np.ones(distances.shape[1])[:, np.newaxis].T * large_value
      new_row[:, all_predictions.shape[0]:] = pseudo_distance
      distances = np.vstack((distances, new_row))



    #with np.printoptions(precision=3, suppress=True):
    #  print("Distances")
    #  print(distances)
        
    # measurement_assignment_ids, prediction_assignment_ids  = linear_sum_assignment(distances)
    measurement_assignment_ids, prediction_assignment_ids  = linear_sum_assignment(distances)
    
    # Different cases
    #    A | B
    #    -----
    #    C | D
    for measurement_id, prediction_id in zip(list(measurement_assignment_ids), 
                                             list(prediction_assignment_ids)):
        # A) measurement <-> existing prediction   => add measurement as new observation of the track
        if measurement_id < measurements.shape[0] and prediction_id < predictions.shape[0]:
            id_ = predictions_ids[prediction_id]
            mm_track_id = id_2_mm[id_]
            model_manager.set_track_measurement(mm_track_id, all_measurements[measurement_id])
            # store observation
            track_history[id_].append([time_step, all_measurements[measurement_id]])

            line = np.stack((all_measurements[measurement_id], all_predictions[prediction_id]), axis=0)
            plt.plot(line[:, 0], line[:, 1], c='green')
            
            
        # B) artificial measurement <-> prediction  => delete track
        elif measurement_id >= measurements.shape[0] and prediction_id < predictions.shape[0]:
            print("B")
            id_ = predictions_ids[prediction_id]
            active_ids.remove(id_)
            mm_track_id = id_2_mm[id_]
            # ToDo: fix this
            # model_manager.free(mm_track_id)
            # del mm_2_id[mm_track_id]
            # del id_2_mm[id_]
            
        # C) measurement <-> artificial prediction  => create track     
        elif measurement_id < measurements.shape[0] and prediction_id >= predictions.shape[0]:
            # new global id
            id_ = next(id_seed)
            active_ids.add(id_)
            # new model manger id
            mm_track_id = model_manager.allocate_track()
            id_2_mm[id_] = mm_track_id
            mm_2_id[mm_track_id] = id_

            model_manager.set_track_measurement(mm_track_id, all_measurements[measurement_id])
            # store observation
            track_history[id_].append([time_step, all_measurements[measurement_id]])

        # D) artificial measurement <-> artificial prediction   => do nothing
        elif measurement_id >= measurements.shape[0] and prediction_id >= predictions.shape[0]:
            print("D")  

    plt.legend(loc="upper left")
    plt.savefig('visualizations/matching_visualization_local/{:05d}'.format(time_step))
    plt.clf()

    df_cm = pd.DataFrame(distances, range(distances.shape[0]),
                  range(distances.shape[1]))
    # sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=False)
    # plt.show()
    plt.clf()

#
print('this is the base mode, now you can execute and test everything as if you were in a normal python shell, that executed all commands until now!')
print('matching can be done now!')
code.interact(local=dict(globals(), **locals()))
# finished_trajectories, trajectories = solve_matching_problem(dataset, model_manager)




# type of errors
# error of first kind: track contains multiple particles
num_errors_of_first_kind = 0
for track_id in track_history.keys():
	track_values = track_history[track_id]
	num_errors_of_first_kind += int(len(list(filter(lambda x: x != track_values[0], track_values))) != 0)

ratio_error_of_first_kind = num_errors_of_first_kind / len(track_history.keys())
print('ratio_error_of_first_kind: ' + str(ratio_error_of_first_kind))

# error of first kind: particle is in multiple tracks
# assigment_of(particles[0][0])
def assigment_of(particle):
	for track_id in track_history.keys():
		track = track_history[track_id]
		for idx in range(len(track)):
			# check if values and time are the same
			if track[idx] == particle[0] and track[idx] == particle[1]:
				return track_id
	print('no matching track id found in assigment_of!')
	code.interact(local=dict(globals(), **locals()))

# get the particles from the dataset instance
particles = dataset.get_particles()
# count the number of errors
num_errors_of_second_kind = 0
for particle in particles:
	reference = assigment_of(particle[0])
	num_errors_of_second_kind += int(len(list(filter(lambda x: assigment_of(x) != reference, particle))) != 0)

ratio_errors_of_second_kind = num_errors_of_second_kind / len(particles)
print('ratio_errors_of_second_kind: ' + str(ratio_errors_of_second_kind))
code.interact(local=dict(globals(), **locals()))
















'''# TODO compare trajectories
# Assumption: We create the trajectories in the same order as they were generated
trajectory_distance_sum = 0.0
# TODO sophisticated trajectory matching
# finished_trajectory_idxs = np.argsort(np.array(list(zip(list(map(lambda x: x[1], finished_trajectories)), list(map(lambda x: x[0][-1][0], finished_trajectories))))))
finished_trajectory_idxs = range(MAX_NUM_TRAJECTORIES)
# code.interact(local=dict(globals(), **locals()))
for i in range(MAX_NUM_TRAJECTORIES):
	for j in range(TRAJECTORY_LENGTH):
		gt = np.array([trajectories[i][1][j], trajectories[i][2][j]])
		trajectory_distance_sum += np.linalg.norm(finished_trajectories[finished_trajectory_idxs[i]][0][j] - gt)
trajectory_distance_avg = trajectory_distance_sum / (MAX_NUM_TRAJECTORIES * TRAJECTORY_LENGTH)
print('data association finished!')
code.interact(local=dict(globals(), **locals()))'''

def solve_matching_problem(dataset, model_manager):
	# solve the actual matching problem
	finished_trajectories = []
	current_trajectories = []
	# maps trajectory ID to trajectory values and trajectory starting point
	trajectories = {}
	# the loop over the time steps
	for it in range(TIME_STEPS):
		print('timestep: ' + str(it))
		# handle different cases based on the result of the matchting
		# case 1 - all predicted measurements can be found in real measurements
		# mask for measurements of current timestep
		current_measurements_real = list(dataset.get_measurement_at_timestep(it))
		current_measurements_real = list(filter(lambda x: x[0] != NAN_VALUE, current_measurements_real))
		num_measurements_real = len(current_measurements_real)
		current_predictions_real = list(model_manager.predict())
		current_predictions_real = list(filter(lambda x: x[0] >= 0, current_predictions_real)) # TODO this is only a heuristic and no guarantee!!!
		num_predictions_real = len(current_predictions_real)
		#
		if num_measurements_real == 0 and num_predictions_real == 0:
			continue
		# case 2 - particle leaves perception
		# the predicted particles which are in the terminal region of the belt
		last_tenth = list(filter(lambda x: SIZE_X - x[0] <= X_DETECTION_TOLERANCE, current_predictions_real))
		last_tenth_prediction_idxss = list(filter(lambda x: SIZE_X - current_predictions_real[x][0] <= X_DETECTION_TOLERANCE, range(num_predictions_real)))
		# for every predicted particle at the end of the belt: add an unique artificial measurement outside of the belt
		current_measurements_artificial = list(map(lambda x: np.array([SIZE_X + X_THRESHOLD, x[1]]), last_tenth))
		
		# case 3 - particle enters perception
		first_tenth = list(filter(lambda x: x[0] <= X_DETECTION_TOLERANCE, current_measurements_real))
		first_tenth_measurement_idxss = list(filter(lambda x: current_measurements_real[x][0] <= X_DETECTION_TOLERANCE, range(num_measurements_real)))
		# for every measurement in the beginning of the belt: add an artificial track
		current_predictions_artificial = list(map(lambda x: np.array([-X_THRESHOLD, x[1]]), first_tenth))
		
		# TODO case 4

		# TODO case 5

		# create the distance matrix
		# Attention: The order of the matrix is used for the classification of event later on
		#	... for example whether the match is an update of a trajectory or a deletion.
		current_measurements = current_measurements_real + current_measurements_artificial
		current_predictions = current_predictions_real + current_predictions_artificial
		# code.interact(local=dict(globals(), **locals()))
		
		# insert the l2 norm between measurement and prediction
		distances = np.zeros([len(current_measurements), len(current_predictions)])
		#print('in solve_matching_problem')
		#code.interact(local=dict(globals(), **locals()))
		for measurement_idxs, measurement in enumerate(current_measurements):
			for prediction_idxs, prediction in enumerate(current_predictions):
				distances[measurement_idxs][prediction_idxs] = np.linalg.norm(measurement - prediction)
		# actual matching -> rows: list() and cols: list()
		# example: rows[0] and cols[0] contain the indices of the best match, ...
		# TODO linear assignment
		# measurement_idxs, prediction_idxs = linear_sum_assignment(distances)
		# current solution with the nearest neighbour approach
		if len(current_measurements) == 0 or len(current_predictions) == 0:
			print('strange behaviour in solve_matching_problem loop')
			code.interact(local=dict(globals(), **locals()))
		measurement_idxs, prediction_idxs = nearest_neighbour(distances)
		elements_to_delete = []
		for i in range(len(measurement_idxs)):
			if measurement_idxs[i] < num_measurements_real and prediction_idxs[i] < num_predictions_real:
				'''if current_measurements[measurement_idxs[i]][1] != current_predictions[prediction_idxs[i]][1]:
					print('something went wrong in update_trajectory!')
					code.interact(local=dict(globals(), **locals()))
				if current_measurements[measurement_idxs[i]][1] != current_trajectories[prediction_idxs[i]][-1][1]: # TODO how to manage mapping from ID to trajectory???
					print("something went wrong inside update_trajectory - type 2")
					code.interact(local=dict(globals(), **locals()))'''
				try:
					trajectory_id = current_trajectories[prediction_idxs[i]]
					model_manager.set_track_measurement(trajectory_id, current_measurements[measurement_idxs[i]])
					# TODO check whether this is correct
					trajectories[trajectory_id][0].append(current_measurements[measurement_idxs[i]])
					trajectories[trajectory_id][1].append(it)
				except Exception as exp:
					print('error in update_trajectory')
					code.interact(local=dict(globals(), **locals()))
			elif measurement_idxs[i] >= num_measurements_real and prediction_idxs[i] >= num_predictions_real:
				# Artificial measurement was matched with artificial prediction
				# that this case won't appear seems to be very problematic for integer programming
				pass
			elif measurement_idxs[i] >= num_measurements_real and prediction_idxs[i] < num_predictions_real:
				# delete the trajectory
				# code.interact(local=dict(globals(), **locals()))
				trajectory_id = current_trajectories[prediction_idxs[i]]
				model_manager.free(trajectory_id)
				finished_trajectories.append(trajectory_id)
				current_trajectories.remove(trajectory_id)
			elif measurement_idxs[i] < num_measurements_real and prediction_idxs[i] >= num_predictions_real:
				# create a new trajetory
				# code.interact(local=dict(globals(), **locals()))
				try:
					trajectory_id = model_manager.allocate_track()
					current_trajectories.append(trajectory_id)
					# TODO check if this really is correct!
					trajectories[trajectory_id] = [[current_measurements[first_tenth_measurement_idxss[prediction_idxs[i] - num_predictions_real]]], [it]]
				except Exception as exp:
					print('error in new_trajectory')
					code.interact(local=dict(globals(), **locals()))
	return finished_trajectories, trajectories