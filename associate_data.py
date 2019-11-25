import numpy as np
from scipy.optimize import linear_sum_assignment
import code

# TODO actual data loading

# external parameters
TIME_STEPS = 100
MAX_NUM_TRAJECTORIES = 50
SIZE_Y = 1000
SIZE_X = 2000
X_DETECTION_TOLERANCE = 10
X_GENERATION_TOLERANCE = 0
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

# solve the actual matching problem
finished_trajectories = []
model = Model()
# the loop over the time steps
for it in range(100):
	# case 1 - all predicted measurements can be found in real measurements
	
	# mask for measurements of current timestep 
	active_indices = list(filter(lambda x: data[it][x][0] != np.Infinity, range(MAX_NUM_TRAJECTORIES)))
	# apply mask
	# list(map(lambda x: data[it - 1][x], list(filter(lambda x: data[it - 1][x][0] != np.Infinity, range(MAX_NUM_TRAJECTORIES)))))
	current_measurements = list(map(lambda x: data[it][x], active_indices))
	num_measurements = len(current_measurements)
	current_predictions = model.predict()
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
			if current_measurements[measurement_idxs[i]][1] != model.current_trajectories[prediction_idxs[i]][-1][1]:
				print("something went wrong inside update_trajectory - type 2")
				code.interact(local=dict(globals(), **locals()))
			try:
				model.update_trajectory(prediction_idxs[i], current_measurements[measurement_idxs[i]], current_predictions[prediction_idxs[i]])
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
				model.new_trajectory(i, current_measurements[first_tenth_measurement_idxss[prediction_idxs[i] - num_predictions]])
			except Exception as exp:
				print('error in new_trajectory')
				code.interact(local=dict(globals(), **locals()))
	# avoid translation of the indexes
	elements_to_delete = np.sort(elements_to_delete)[::-1]
	for idx in range(elements_to_delete.shape[0]):
		if len(model.current_trajectories[elements_to_delete[idx]]) != TRAJECTORY_LENGTH:
			print('trajectory length missmatch')
			code.interact(local=dict(globals(), **locals()))
		try:
			finished_trajectories.append(model.delete_trajectory(elements_to_delete[idx]))
		except Exception as exp:
			print('error in delete_trajectory')
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
