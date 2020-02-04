import code  # code.interact(local=dict(globals(), **locals()))
import logging


class Evaluator(object):
    def __init__(self, global_config, particles, tracks):
        self.global_config = global_config
        self.particles = particles
        self.tracks = tracks

        self.particles_hash_map = self._create_particles_hash_map()
        self.tracks_hash_map = self._create_tracks_hash_map()

    def _create_particles_hash_map(self):
        # dict(hash(str([0.1, 0.2])) -> particle_id)
        particles_hash_map = {}

        for idx, particle_list in enumerate(self.particles):
            for particle_timestep, particle_measurement in particle_list:
                hash_ = str(particle_measurement)
                if hash_ in particles_hash_map:
                    raise ValueError("Duplicate particle measurements")
                particles_hash_map[hash_] = idx

        return particles_hash_map

    def _create_tracks_hash_map(self):
        # dict(hash(str([0.1, 0.2])) -> track_id)
        tracks_hash_map = {}

        for idx in list(self.tracks.keys()):
            track = self.tracks[idx]
            for it, track_measurement in enumerate(track.measurements):
                hash_ = str(track_measurement)+"|"+str(it + track.initial_timestep)
                if hash_ in tracks_hash_map:
                    raise ValueError("Duplicate track measurements")
                tracks_hash_map[hash_] = idx

        return tracks_hash_map

    def assigment_of_measurement_in_particle(self, timestep, measurement):
        try:
            return self.particles_hash_map[str(measurement)]
        except KeyError:
            # logging.debug('Measurement not found in particles hash map. Probably an artificial measurement.')
            return -1

    def assigment_of_measurement_in_track(self, timestep, measurement):
        try:
            return self.tracks_hash_map[str(measurement)+"|"+str(timestep)]
        except KeyError:
            # logging.debug('Measurement not found in track hash map.')
            return -1

    # type of errors
    # error of first kind: track contains multiple particles
    def error_of_first_kind(self):
        num_errors_of_first_kind = 0

        for track_id in list(self.tracks.keys()):
            logging.debug(track_id)
            track = self.tracks[track_id]

            particle_id = self.assigment_of_measurement_in_particle(track.initial_timestep, track.measurements[0])

            for it, measurement in enumerate(track.measurements):
                particle_id_current = self.assigment_of_measurement_in_particle(track.initial_timestep + it, measurement)
                if particle_id_current != particle_id and particle_id_current != -1:
                    num_errors_of_first_kind += 1
                    break

        ratio_error_of_first_kind = num_errors_of_first_kind / len(self.tracks)
        logging.info('ratio_error_of_first_kind: ' + str(ratio_error_of_first_kind))
        return ratio_error_of_first_kind

    def error_of_second_kind(self):
        num_errors_of_second_kind = 0

        for particle_id, particle_list in enumerate(self.particles):
            track_id = self.assigment_of_measurement_in_track(particle_list[0][0], particle_list[0][1])
            logging.debug(track_id)

            for it, particle in enumerate(particle_list):
                track_id_current = self.assigment_of_measurement_in_track(particle[0], particle[1])
                if track_id_current != track_id and track_id_current != -1:
                    num_errors_of_second_kind += 1
                    break

        ratio_error_of_second_kind = num_errors_of_second_kind / len(self.particles)
        logging.info('ratio_error_of_second_kind: {}'.format(ratio_error_of_second_kind))
        return ratio_error_of_second_kind
