"""Track Manager.

Change log (Please insert your name here if you worked on this file)
    * Created by: Daniel Pollithy 
    * Jakob Thumm (jakob.thumm@student.kit.edu) 2.10.2020:    Completed documentation.
"""
import logging
from model_manager import ModelManager
from track import Track
import code

class TrackManager(object):
    """The track manager handels the birth, update and death of all tracks."""

    def __init__(self, track_config):
        """Create a TrackManager object.
        
        Args:
            track_config (dict):    Default values for the new tracks.
        """
        self.tracks = {}
        self.active_ids = []
        self.currently_highest_id = -1
        self.track_config = track_config

    def real_track_real_measurement(self, global_track_id, particle_id, measurement, model_manager):
        """Add a real measurement to an existing track.

        Args:
            global_track_id (int):          Track ID
            particle_id (int):              Particle ID
            measurement (array):            The [x, y] coordinates of the particle
            model_manager (ModelManager):   ModelManager object needed to call model_manager.update_by_id()
        """
        self.tracks[global_track_id].add_measurement(measurement=measurement, particle_id=particle_id, is_artificial=False)
        model_manager.update_by_id(global_track_id, measurement)

    def real_track_pseudo_measurement(self, global_track_id, measurement, model_manager):
        """Add a fake measurement to an existing track.

        Args:
            global_track_id (int):          Track ID
            particle_id (int):              Particle ID (default value)
            measurement (array):            The [x, y] coordinates of the particle (only prediction, no real measurement)
            model_manager (ModelManager):   ModelManager object needed to call model_manager.update_by_id()
        """
        is_alive_probability = self.tracks[global_track_id].add_measurement(measurement=measurement, is_artificial=True)
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

    def pseudo_track_real_measurement(self, measurement, particle_id, current_timestep, model_manager):
        """Create a new track and add a real measurement.

        Increase the global track ID

        Args:
            particle_id (int):              Particle ID (default value)
            measurement (array):            The [x, y] coordinates of the particle (only prediction, no real measurement)
            current_timestep (int):         Current time step of particle
            model_manager (ModelManager):   ModelManager object needed to call model_manager.update_by_id()

        Returns:
            updated global_track_id
        """
        self.currently_highest_id += 1
        global_track_id = self.currently_highest_id
        self.active_ids.append(global_track_id)
        self.tracks[global_track_id] = Track(current_timestep, measurement, particle_id, **self.track_config)
        model_manager.create_by_id(global_track_id, measurement)
        return global_track_id

    def get_predictions(self, model_manager):
        """Call and return model_manager.predict_all()."""
        predictions = model_manager.predict_all()
        if len(self.active_ids) != len(predictions.keys()):
            logging.error("something with the id management doesnt work in get_predictions!")
            code.interact(local=dict(globals(), **locals()))
        return predictions

    def get_alive_probability(self, track_id):
        """Return the probability that track is alive."""
        return self.tracks[track_id].is_alive_probability

    def get_highest_track_id(self):
        """Return the current highest track ID."""
        return self.currently_highest_id

    def get_tracks(self):
        """Return track dictionary."""
        return self.tracks
