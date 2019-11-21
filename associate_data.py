import numpy as np
from scipy.optimize import linear_sum_assignment
import code

# TODO actual data loading

# external parameters
TIME_STEPS = 100
MAX_NUM_TRAJECTORIES = 50
SIZE_Y = 1000
SIZE_X = 2000
TRAJECTORY_LENGTH = 15

# generate artificial data
# assumed basic structure of data
data = np.ones([TIME_STEPS, MAX_NUM_TRAJECTORIES, 2], dtype=np.float32) * np.Infinity

# generate artificial trajectories
trajectories = []
#
for i in range(MAX_NUM_TRAJECTORIES):
	start_frame = np.random.randint(TIME_STEPS - TRAJECTORY_LENGTH)
	start_x = np.random.randint(0, MAX_NUM_TRAJECTORIES)
	start_y = np.random.randint(SIZE_Y)
	end_x = np.random.randint(SIZE_X - MAX_NUM_TRAJECTORIES, SIZE_X)
	end_y = np.random.randint(SIZE_Y)
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
		self.current_trajectories = []

	def predict(self):
		predictions = []
		for trajectory in self.current_trajectories:
			if len(trajectory) == 1:
				new_x = trajectory[0][0] + (SIZE_X/TRAJECTORY_LENGTH)
				new_y = trajectory[0][1]
			else:
				new_x = 2 * trajectory[-1][0] - trajectory[-2][0]
				new_y = 2 * trajectory[-1][1] - trajectory[-2][1]
			predictions.append(np.array([new_x, new_y]))
		return predictions

	def new_trajectory(self, initial_value):
		self.current_trajectories.append([initial_value])

	def update_trajectory(self, index, measured_value, predicted_value):
		self.current_trajectories[index].append(0.5 * (measured_value + predicted_value)) # TODO how to do this properly???

	def delete_trajectory(self, index):
		return self.current_trajectories.pop(index)

# solve the actual matching problem
finished_trajectories = []
model = Model()
# the loop over the time steps
for it in range(100):
	# case 1
	active_indices = list(filter(lambda x: data[it][x][0] != np.Infinity, range(MAX_NUM_TRAJECTORIES)))
	current_measurements = list(map(lambda x: data[it][x], active_indices))
	num_measurements = len(current_measurements)
	current_predictions = model.predict()
	num_predictions = len(current_predictions)
	# case 2 - particle leaves perception
	last_tenth = list(filter(lambda x: SIZE_X - x[0] <= 200, current_predictions))
	last_tenth_idxs = list(filter(lambda x: SIZE_X - current_predictions[x][0] <= 200, range(num_predictions)))
	last_tenth_artificial_measurements = list(map(lambda x: np.array([2005, x[1]]), last_tenth))
	# case 3 - particle enters perception
	first_tenth = list(filter(lambda x: x[0] <= 200, current_measurements))
	first_tenth_idxs = list(filter(lambda x: current_measurements[x][0] <= 200, range(num_measurements)))
	last_tenth_artificial_predictions = list(map(lambda x: np.array([-5, x[1]]), first_tenth))
	# TODO case 4

	# TODO case 5

	# create the distance matrix
	current_measurements += last_tenth_artificial_measurements
	current_predictions += last_tenth_artificial_predictions
	# code.interact(local=dict(globals(), **locals()))
	distances = np.zeros([len(current_measurements), len(current_predictions)])
	for measurement_idx, measurement in enumerate(current_measurements):
		for prediction_idx, prediction in enumerate(current_predictions):
			distances[measurement_idx][prediction_idx] = np.linalg.norm(measurement - prediction)
	# actual matching
	row_ind, col_ind = linear_sum_assignment(distances)
	for i in range(len(row_ind)):
		if row_ind[i] < num_measurements and col_ind[i] < num_predictions:
			model.update_trajectory(i, current_measurements[row_ind[i]], current_predictions[col_ind[i]])
		elif row_ind[i] >= num_measurements and col_ind[i] >= num_predictions:
			pass # nothing to do or am i overseeing something?
		elif row_ind[i] >= num_measurements and col_ind[i] < num_predictions:
			try:
				finished_trajectories.append(model.delete_trajectory(last_tenth_idxs[row_ind[i] - num_measurements]))
			except Exception:
				code.interact(local=dict(globals(), **locals()))
		else:
			# code.interact(local=dict(globals(), **locals()))
			model.new_trajectory(current_measurements[first_tenth_idxs[col_ind[i] - num_predictions]])

# TODO compare trajectories
trajectory_distance_sum = 0
for i in len(MAX_NUM_TRAJECTORIES):
	for j in range(TRAJECTORY_LENGTH):
		trajectory_distance_sum += np.linalg.norm(finished_trajectories[i][j] - trajectories[i][j])
trajectory_distance_avg = trajectory_distance_sum / (MAX_NUM_TRAJECTORIES * TRAJECTORY_LENGTH)
print('data association finished!')
code.interact(local=dict(globals(), **locals()))