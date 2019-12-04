import numpy as np
from scipy.optimize import linear_sum_assignment
from data import CsvDataSet, FakeDataSet
from model import ModelManager, rnn_model_factory
import model
import data
import tensorflow as tf
from tensorflow.keras import backend as K
import code

MODE = 'actual_pipeline'

# external parameters
if MODE == 'minimal_example':
	# using my own data loader and my own model
	TIME_STEPS = 100
	MAX_NUM_TRAJECTORIES = 50
	SIZE_Y = 1000
	SIZE_X = 2000
	X_DETECTION_TOLERANCE = 10
	X_GENERATION_TOLERANCE = 0
	NAN_VALUE = np.Infinity
	X_THRESHOLD = 5
	TRAJECTORY_LENGTH = 15

	# generate artificial data
	# assumed basic structure of data
	data = np.ones([TIME_STEPS, MAX_NUM_TRAJECTORIES, 2], dtype=np.float32) * np.Infinity

	# generate artificial trajectories
	trajectories = []
	#
	for i in range(MAX_NUM_TRAJECTORIES):
		start_frame = np.random.randint(TIME_STEPS - TRAJECTORY_LENGTH)
		# start_x = np.random.randint(0, X_GENERATION_TOLERANCE)
		start_x = 0
		start_y = np.random.randint(0, SIZE_Y)
		# end_x = np.random.randint(SIZE_X - X_GENERATION_TOLERANCE, SIZE_X)
		end_x = SIZE_X
		# TODO more complicated case needs better model
		# end_y = np.random.randint(SIZE_Y)
		end_y = start_y
		trajectory_x = start_x + (end_x - start_x) * (1 / (TRAJECTORY_LENGTH - 1)) * np.arange(TRAJECTORY_LENGTH)
		trajectory_y = start_y + (end_y - start_y) * (1 / (TRAJECTORY_LENGTH - 1)) * np.arange(TRAJECTORY_LENGTH)
		# TODO noise over the trajectories
		# TODO drop out part of the trajectory
		trajectories.append([start_frame, trajectory_x, trajectory_y])
		for j in range(TRAJECTORY_LENGTH):
			data[start_frame + j][i][0] = trajectory_x[j]
			data[start_frame + j][i][1] = trajectory_y[j]

	# TODO connection to the actual model

	# a naive model based on linear regression and average x step sizes
	class Model:
		def __init__(self):
			self.trajectory_start_frames = []
			self.current_trajectories = []

		def predict(self):
			predictions = []
			for trajectory in self.current_trajectories:
				if len(trajectory) == 1:
					new_x = trajectory[0][0] + (SIZE_X / (TRAJECTORY_LENGTH - 1))
					new_y = trajectory[0][1]
				else:
					new_x = 2 * trajectory[-1][0] - trajectory[-2][0]
					new_y = 2 * trajectory[-1][1] - trajectory[-2][1]
				if new_y != trajectory[-1][1]:
					print("something went wrong inside predict")
					code.interact(local=dict(globals(), **locals()))
				predictions.append(np.array([new_x, new_y]))
			return predictions

		def new_trajectory(self, start_frame, initial_value):
			self.trajectory_start_frames.append(start_frame)
			self.current_trajectories.append([initial_value])

		def update_trajectory(self, index, measured_value, predicted_value):
			if measured_value[1] != predicted_value[-1]:
				print("something went wrong inside update_trajectory")
				code.interact(local=dict(globals(), **locals()))
			# self.current_trajectories[index].append(0.5 * (measured_value + predicted_value)) # TODO how to do this properly???
			self.current_trajectories[index].append(measured_value) # TODO how to do this properly???

		def delete_trajectory(self, index):
			return self.current_trajectories.pop(index), self.trajectory_start_frames.pop(index)

	# create the actual model
	model_manager = Model()
else:
	# using the external data set loader and an actual model
	num_time_steps = 35
	batch_size = 128
	NAN_VALUE = np.Infinity
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
	model_manager = ModelManager(dataset.n_trajectories, 16, rnn_model)

	#
	TIME_STEPS = num_time_steps

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
	track_ids = []
	# the loop over the time steps
	for it in range(TIME_STEPS):
		print('timestep: ' + it)
		# case 1 - all predicted measurements can be found in real measurements
		# mask for measurements of current timestep
		if MODE == 'minimal_example':
			active_indices = list(filter(lambda x: data[it][x][0] != NAN_VALUE, range(MAX_NUM_TRAJECTORIES)))
			current_measurements = list(map(lambda x: data[it][x], active_indices))
		else:
			current_measurements = dataset.get_measurement_at_timestep(it)
		num_measurements = len(current_measurements)
		if MODE == 'minimal_example':
			current_trajectories = model_manager.current_trajectories
			current_predictions = model_manager.predict()
		else:
			current_predictions = list(model_manager.predict())
		num_predictions = len(current_predictions)
		#
		if num_measurements == 0 and num_predictions == 0:
			continue
		# case 2 - particle leaves perception
		# the predicted particles which are in the terminal region of the belt
		last_tenth = list(filter(lambda x: SIZE_X - x[0] <= X_DETECTION_TOLERANCE, current_predictions))
		last_tenth_prediction_idxss = list(filter(lambda x: SIZE_X - current_predictions[x][0] <= X_DETECTION_TOLERANCE, range(num_predictions)))
		# for every predicted particle at the end of the belt: add an unique artificial measurement outside of the belt
		last_tenth_artificial_measurements = list(map(lambda x: np.array([SIZE_X + X_THRESHOLD, x[1]]), last_tenth))
		
		# case 3 - particle enters perception
		first_tenth = list(filter(lambda x: x[0] <= X_DETECTION_TOLERANCE, current_measurements))
		first_tenth_measurement_idxss = list(filter(lambda x: current_measurements[x][0] <= X_DETECTION_TOLERANCE, range(num_measurements)))
		# for every measurement in the beginning of the belt: add an artificial track
		first_tenth_artificial_predictions = list(map(lambda x: np.array([-X_THRESHOLD, x[1]]), first_tenth))
		
		# TODO case 4

		# TODO case 5

		# create the distance matrix
		# Attention: The order of the matrix is used for the classification of event later on
		#  ... for example whether the match is an update of a trajectory or a deletion.
		current_measurements += last_tenth_artificial_measurements
		current_predictions += first_tenth_artificial_predictions
		# code.interact(local=dict(globals(), **locals()))
		
		# insert the l2 norm between measurement and prediction
		distances = np.zeros([len(current_measurements), len(current_predictions)])
		for measurement_idxs, measurement in enumerate(current_measurements):
			for prediction_idxs, prediction in enumerate(current_predictions):
				distances[measurement_idxs][prediction_idxs] = np.linalg.norm(measurement - prediction)
		# actual matching -> rows: list() and cols: list()
		# example: rows[0] and cols[0] contain the indices of the best match, ...
		# TODO linear assignment
		# measurement_idxs, prediction_idxs = linear_sum_assignment(distances)
		# current solution with the nearest neighbour approach
		if len(current_measurements) == 0 or len(current_predictions) == 0:
			print('strange behaviour')
			code.interact(local=dict(globals(), **locals()))
		measurement_idxs, prediction_idxs = nearest_neighbour(distances)
		elements_to_delete = []
		for i in range(len(measurement_idxs)):
			if measurement_idxs[i] < num_measurements and prediction_idxs[i] < num_predictions:
				if current_measurements[measurement_idxs[i]][1] != current_predictions[prediction_idxs[i]][1]:
					print('something went wrong in update_trajectory!')
					code.interact(local=dict(globals(), **locals()))
				if current_measurements[measurement_idxs[i]][1] != current_trajectories[prediction_idxs[i]][-1][1]: # TODO how to manage mapping from ID to trajectory???
					print("something went wrong inside update_trajectory - type 2")
					code.interact(local=dict(globals(), **locals()))
				try:
					if MODE == 'minimal_example':
						model_manager.update_trajectory(prediction_idxs[i], current_measurements[measurement_idxs[i]], current_predictions[prediction_idxs[i]])
					else:
						# TODO debug
						model_manager.set_track_measurement((prediction_idxs[i], current_measurements[measurement_idxs[i]]))
				except Exception as exp:
					print('error in update_trajectory')
					code.interact(local=dict(globals(), **locals()))
			elif measurement_idxs[i] >= num_measurements and prediction_idxs[i] >= num_predictions:
				# Artificial measurement was matched with artificial prediction
				# that this case won't appear seems to be very problematic for integer programming
				pass # nothing to do or am i overseeing something?
			elif measurement_idxs[i] >= num_measurements and prediction_idxs[i] < num_predictions:
				#print('see what happens before trajectory gets deleted')
				#code.interact(local=dict(globals(), **locals()))
				elements_to_delete.append(prediction_idxs[i])
			elif measurement_idxs[i] < num_measurements and prediction_idxs[i] >= num_predictions:
				# code.interact(local=dict(globals(), **locals()))
				try:
					if MODE == 'minimal_example':
						model_manager.new_trajectory(i, current_measurements[first_tenth_measurement_idxss[prediction_idxs[i] - num_predictions]])
					else:
						# TODO debug
						current_trajetories.append(model_manager.allocate_track())
				except Exception as exp:
					print('error in new_trajectory')
					code.interact(local=dict(globals(), **locals()))
		# avoid translation of the indexes
		elements_to_delete = np.sort(elements_to_delete)[::-1]
		for idx in range(elements_to_delete.shape[0]):
			if len(current_trajectories[elements_to_delete[idx]]) != TRAJECTORY_LENGTH: # TODO how to manage mapping from ID to trajectory???
				print('trajectory length missmatch')
				code.interact(local=dict(globals(), **locals()))
			try:
				if MODE == 'minimal_example':
					finished_trajectories.append(model_manager.delete_trajectory(elements_to_delete[idx]))
				else:
					# TODO debug
					model_manager.free(elements_to_delete[idx])
					current_trajectories.pop(elements_to_delete[idx])
			except Exception as exp:
				print('error in delete_trajectory')
				code.interact(local=dict(globals(), **locals()))
	return finished_trajectories
#
print('matching can be done now!')
code.interact(local=dict(globals(), **locals()))
finished_trajectories = solve_matching_problem(dataset, model_manager)


# type of errors
# track contains multiple particles
num_errors_of_first_kind = 0
for track in finished_trajectories:
	num_errors_of_first_kind += int(len(list(filter(lambda x: x != track[0], track))) != 0)
# particle is in multiple tracks
num_errors_of_second_kind = 0
# TODO get particles
for particle in particles:
	# TODO define assigment_of(particle)
	num_errors_of_second_kind += int(len(list(filter(lambda x: assigment_of(x) != assigment_of(particle[0]), particle))) != 0)
code.interact(local=dict(globals(), **locals()))

# TODO compare trajectories
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
code.interact(local=dict(globals(), **locals()))