"""Track class for Multitarget Tracking

Change log (Please insert your name here if you worked on this file)
    * Created by: Daniel Pollithy 
    * Jakob Thumm (jakob.thumm@student.kit.edu) 2.10.2020:    Completed documentation.
"""
import numpy as np


class Track(object):
    """The Track class for multitarget tracking."""

    def __init__(self, initial_timestep, first_measurement, first_particle_id,
                 initial_is_alive_probability=0.5,
                 is_alive_increase=0.5,
                 is_alive_decrease=0.25):
        """Create a track.

        Args:
            initial_timestep (int):                 First time step
            first_measurement (array):              First measurement of track
            first_particle_id (int):                ID of associated particle
            initial_is_alive_probability (double):  Probability value that the track is still alive
            is_alive_increase (double):             Increase the alive-probability by this value every time a measurement is associated to the track
            is_alive_decrease (double):             Decrease the alive-probability by this value every time no measurement is associated to the track
        """
        self.measurements = [first_measurement]
        self.particle_ids = [int(first_particle_id)]
        self.initial_timestep = initial_timestep
        self.is_alive_probability = initial_is_alive_probability
        self.is_alive_decrease = is_alive_decrease
        self.is_alive_increase = is_alive_increase

    def add_measurement(self, measurement, particle_id=-1, is_artificial=True):
        """Make a track <-> measurement association.

        Args:
            measurement (array):        New [x, y] coordinates
            particle_id (int):          ID of newly associated particle
            is_artificial (Boolean):    Is the new associated measurement artificial or real?

        Returns:
            is_alive_probability (double)
        """
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
