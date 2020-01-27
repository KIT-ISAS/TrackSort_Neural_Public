import code  # code.interact(local=dict(globals(), **locals()))

class Evaluator(object):
	def __init__(self, global_config, particles, tracks):
		self.global_config = global_config
		self.particles = particles
		self.tracks = tracks

	# type of errors
	# error of first kind: track contains multiple particles
	def assigment_of_measurement_in_particle(self, timestep, measurement):
		for idx, particle_list in enumerate(self.particles):
			for particle in particle_list:
				particle_timestep, particle_measurement = particle
				timestep = particle_timestep # TODO fix particle timestep extraction
				if timestep == particle_timestep and particle_measurement[0] == measurement[0] and particle_measurement[1] == measurement[1]:
					if self.global_config['verbose'] > 0: print(str(timestep) + ' - ' + str(measurement) + ' - measurement found match in particles! - ' + str(idx))
					return idx
		if self.global_config['verbose'] > 0: print(str(timestep) + ' - ' + str(measurement) + ' - measurement found no match in particles! Probably a artificial measurement.')
		# code.interact(local=dict(globals(), **locals()))
		return -1

	def error_of_first_kind(self):
		num_errors_of_first_kind = 0
		for track_id in list(self.tracks.keys()):
			if self.global_config['verbose'] > 0: print(track_id)
			track = self.tracks[track_id]
			# code.interact(local=dict(globals(), **locals()))
			particle_id = self.assigment_of_measurement_in_particle(track.initial_timestep, track.measurements[0])
			def correct_condition(x):
				particle_id_current = self.assigment_of_measurement_in_particle(x[0], x[1])
				return particle_id_current != particle_id and particle_id_current != -1
			check_list = []
			for it, measurement in enumerate(track.measurements):
				check_list.append([track.initial_timestep + it, measurement])
			is_incremented = int(len(list(filter(correct_condition, check_list))) != 0)
			if is_incremented and self.global_config['debug']:
				print('in error_of_first_kind')
				code.interact(local=dict(globals(), **locals()))
			num_errors_of_first_kind += is_incremented
		ratio_error_of_first_kind = num_errors_of_first_kind / len(self.tracks)
		if self.global_config['verbose'] > 0: print('ratio_error_of_first_kind: ' + str(ratio_error_of_first_kind))
		return ratio_error_of_first_kind

	# type of errors
	# error of first kind: track contains multiple particles
	def assigment_of_measurement_in_track(self, timestep, measurement):
		for idx in list(self.tracks.keys()):
			track = self.tracks[idx]
			for it, track_measurement in enumerate(track.measurements):
				track_timestep = track.initial_timestep + it
				timestep = track_timestep # TODO fix particle timestep extraction
				if track_timestep == timestep and track_measurement[0] == measurement[0] and track_measurement[1] == measurement[1]:
					if self.global_config['verbose'] > 0: print(str(timestep) + ' - ' + str(measurement) + ' - measurement found match in particles! - ' + str(idx))
					return idx
		else:
			if self.global_config['verbose'] > 0: print(str(timestep) + ' - ' + str(measurement) + ' - measurement found no match in particles! Probably a artificial measurement.')
			# if self.global_config['verbose'] > 0: print('measurement found no match in tracks!')
			# code.interact(local=dict(globals(), **locals()))
			return -1


	def error_of_second_kind(self):
		num_errors_of_second_kind = 0
		for particle_id, particle_list in enumerate(self.particles):
			'''if particle_id > 100:
				break  # TODO remove this again!'''
			track_id = self.assigment_of_measurement_in_track(particle_list[0][0], particle_list[0][1])
			def correct_condition(x):
				track_id_current = self.assigment_of_measurement_in_track(x[0], x[1])
				return track_id_current != track_id and track_id_current != -1
			check_list = []
			for it, particle in enumerate(particle_list):
				check_list.append([particle[0], particle[1]])
			num_errors_of_second_kind += int(len(list(filter(correct_condition, check_list))) != 0)


		#ratio_error_of_second_kind = num_errors_of_second_kind / len(self.particles)
		ratio_error_of_second_kind = num_errors_of_second_kind / len(self.particles) # TODO remove this again!
		if self.global_config['verbose'] > 0: print('ratio_error_of_second_kind: ' + str(ratio_error_of_second_kind))
		return ratio_error_of_second_kind

