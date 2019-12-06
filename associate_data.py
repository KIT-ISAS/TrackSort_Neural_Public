import numpy as np
from scipy.optimize import linear_sum_assignment
from data import CsvDataSet, FakeDataSet
from model import ModelManager, rnn_model_factory
import model
import data
import tensorflow as tf
from tensorflow.keras import backend as K
import code

# using the external data set loader and an actual model
num_time_steps = 35
batch_size = 128
NAN_VALUE = -1
belt_width = 2000

dataset = data.FakeDataSet(timesteps=num_time_steps, 
							batch_size=batch_size,
							number_trajectories=6000, 
							additive_noise_stddev=2, 
							additive_target_stddev=20,
							belt_width=belt_width)
#
dataset_train, dataset_test = dataset.get_tf_data_sets_seq2seq_data(normalized=True)

# Train model
rnn_model, hash_ = model.rnn_model_factory(
		num_units_first_rnn=2, 
		num_units_second_rnn=0,
		num_units_first_dense=0,
		rnn_model_name='lstm',
		num_time_steps=num_time_steps, 
		batch_size=batch_size, 
		nan_value=NAN_VALUE, 
		# unroll=False,
		input_dim=2)
print(rnn_model.summary())

optimizer = tf.keras.optimizers.Adam()
train_step_fn = model.train_step_generator(rnn_model, optimizer)
calc_mae_test_fn = model.tf_error(rnn_model, dataset_test, belt_width, squared=False)

loss_history = []
total_num_epochs = 400

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
		loss_history.append(loss)

	print("{}/{}: \t loss={}".format(epoch, total_num_epochs, loss))

# test model
test_mae = calc_mae_test_fn()
print(test_mae.numpy())

# create the model manager
model_manager = ModelManager(dataset.n_trajectories, batch_size, rnn_model)

#
TIME_STEPS = num_time_steps
MAX_NUM_TRAJECTORIES = dataset.n_trajectories
SIZE_X = dataset.belt_max_x
SIZE_Y = dataset.belt_max_y
# the region, where the track is considered to appear / disappear
X_DETECTION_TOLERANCE = SIZE_X / 10
# the distance of the pseudo measurements and tracks to the actual belt
X_THRESHOLD = 5


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
#
print('this is the base mode, now you can execute and test everything as if you were in a normal python shell, that executed all commands until now!')
print('matching can be done now!')
code.interact(local=dict(globals(), **locals()))
finished_trajectories, trajectories = solve_matching_problem(dataset, model_manager)


# type of errors
# error of first kind: track contains multiple particles
num_errors_of_first_kind = 0
for track_id in finished_trajectories:
	track = trajectories[track_id]
	track_values = track[0]
	num_errors_of_first_kind += int(len(list(filter(lambda x: x != track_values[0], track_values))) != 0)
ratio_error_of_first_kind = num_errors_of_first_kind / len(finished_trajectories)
print('ratio_error_of_first_kind: ' + str(ratio_error_of_first_kind))

# error of first kind: particle is in multiple tracks
def assigment_of(particle):
	for track_id in finished_trajectories:
		track = trajectories[track_id]
		for idx in range(len(track[0])):
			# check if values and time are the same
			if track[0][idx] == particle[0] and track[1][idx] == particle[1]:
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