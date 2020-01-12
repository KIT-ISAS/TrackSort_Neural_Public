from model_manager import ModelManager
from track import Track
import code  # code.interact(local=dict(globals(), **locals()))


class TrackManager(object):
    def __init__(self, global_config, data_source):
        self.global_config = global_config
        self.model_manager = ModelManager(global_config, data_source)
        self.tracks = {}
        self.active_ids = []
        self.currently_highest_id = 0

    def real_track_real_measurement(self, global_track_id, measurement):
        self.tracks[global_track_id].add_measurement(measurement, is_artificial=False)
        self.model_manager.update_by_id(global_track_id, measurement)

    def real_track_pseudo_measurement(self, global_track_id, measurement):
        is_alive_probability = self.tracks[global_track_id].add_measurement(measurement, is_artificial=True)
        if is_alive_probability >= 0:
            self.model_manager.update_by_id(global_track_id, measurement)
            return True
        else:
            print('in real_track_pseudo_measurement')
            # code.interact(local=dict(globals(), **locals()))
            try:
                self.active_ids.remove(global_track_id)
            except Exception:
                print('error in real_track_pseudo_measurement')
                code.interact(local=dict(globals(), **locals()))
            self.model_manager.delete_by_id(global_track_id)
            return False

    def pseudo_track_real_measurement(self, measurement, current_timestep):
        global_track_id = self.currently_highest_id
        self.global_config['highest_id'] = global_track_id
        self.currently_highest_id += 1
        self.active_ids.append(global_track_id)
        self.tracks[global_track_id] = Track(self.global_config, current_timestep, measurement,
                                             **self.global_config['Track'])
        self.model_manager.create_by_id(global_track_id, measurement)
        return global_track_id

    def get_predictions(self):
        predictions = self.model_manager.predict_all()
        if len(self.active_ids) != len(predictions.keys()):
            print("something with the id management doesnt work in get_predictions!")
            code.interact(local=dict(globals(), **locals()))
        return predictions
