"""Track.

TODO:
    * Add docstring
"""
import numpy as np


class Track(object):
    def __init__(self, initial_timestep, first_measurement, first_particle_id,
                 initial_is_alive_probability=0.5,
                 is_alive_increase=0.5,
                 is_alive_decrease=0.25):
        self.measurements = [first_measurement]
        self.particle_ids = [int(first_particle_id)]
        self.initial_timestep = initial_timestep
        self.is_alive_probability = initial_is_alive_probability
        self.is_alive_decrease = is_alive_decrease
        self.is_alive_increase = is_alive_increase

    def add_measurement(self, measurement, particle_id=-1, is_artificial=True):
        self.measurements.append(measurement)
        if is_artificial:
            self.is_alive_probability -= self.is_alive_decrease
        else:
            self.is_alive_probability = min(1.0, self.is_alive_probability + self.is_alive_increase)
            self.particle_ids.append(int(particle_id))
        return self.is_alive_probability

    def get_unique_particle_ids(self):
        """Return a numpy array of unique particle ids."""
        return np.unique(self.particle_ids)
