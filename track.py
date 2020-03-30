"""Track.

TODO:
    * Add docstring
"""

class Track(object):
    def __init__(self, initial_timestep, first_measurement,
                 initial_is_alive_probability=0.5,
                 is_alive_increase=0.5,
                 is_alive_decrease=0.25):
        self.measurements = [first_measurement]
        self.initial_timestep = initial_timestep
        self.is_alive_probability = initial_is_alive_probability
        self.is_alive_decrease = is_alive_decrease
        self.is_alive_increase = is_alive_increase

    def add_measurement(self, measurement, is_artificial):
        self.measurements.append(measurement)
        if is_artificial:
            self.is_alive_probability -= self.is_alive_decrease
        else:
            self.is_alive_probability = min(1.0, self.is_alive_probability + self.is_alive_increase)
        return self.is_alive_probability
