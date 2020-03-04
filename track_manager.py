import logging

from model_manager import ModelManager
from track import Track
import code


class TrackManager(object):
    def __init__(self, global_config):
        self.global_config = global_config
        self.tracks = {}
        self.active_ids = []
        self.currently_highest_id = 0

    def real_track_real_measurement(self, global_track_id, measurement, model_manager):
        """ 
        Add the given measurement to an existing track
        """
        self.tracks[global_track_id].add_measurement(measurement, is_artificial=False)
        model_manager.update_by_id(global_track_id, measurement)

    def real_track_pseudo_measurement(self, global_track_id, measurement, model_manager):
        is_alive_probability = self.tracks[global_track_id].add_measurement(measurement, is_artificial=True)
        if is_alive_probability >= 0:
            model_manager.update_by_id(global_track_id, measurement)
            return True
        else:
            try:
                self.active_ids.remove(global_track_id)
            except Exception:
                logging.error('error in real_track_pseudo_measurement')
                code.interact(local=dict(globals(), **locals()))
            model_manager.delete_by_id(global_track_id)
            return False

    def pseudo_track_real_measurement(self, measurement, current_timestep, model_manager):
        """ 
        Create a new track
        Add the given measurement to a new track and increases the global_track_id
        Return the global_track_id
        """
        global_track_id = self.currently_highest_id
        self.global_config['highest_id'] = global_track_id
        self.currently_highest_id += 1
        self.active_ids.append(global_track_id)
        self.tracks[global_track_id] = Track(self.global_config, current_timestep, measurement,
                                             **self.global_config['Track'])
        model_manager.create_by_id(global_track_id, measurement)
        return global_track_id

    def get_predictions(self, model_manager):
        predictions = model_manager.predict_all()
        if len(self.active_ids) != len(predictions.keys()):
            logging.error("something with the id management doesnt work in get_predictions!")
            code.interact(local=dict(globals(), **locals()))
        return predictions

    def get_alive_probability(self, track_id):
        return self.tracks[track_id].is_alive_probability
