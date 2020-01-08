class Track(object):
    def __init__(self, global_config, initial_timestep, first_measurement):
        self.global_config = global_config
        self.measurements = [first_measurement]
        self.initial_timestep = initial_timestep
        self.is_alive_probability = self.global_config['initial_is_alive_probability']

    def add_measurement(self, measurement, is_artificial):
        self.measurements.append(measurement)
        if is_artificial:
            self.is_alive_probability -= self.global_config['is_alive_decrease']
        else:
            self.is_alive_probability = min(1.0, self.is_alive_probability + self.global_config['is_alive_increase'])
        return self.is_alive_probability
