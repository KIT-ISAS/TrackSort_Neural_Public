"""KF Model and KF State.

Todo:
    * Implement CA Model (and more)
    * (Convert np representation to tensor representation for mixture of experts)
"""

import numpy as np
import pickle
import logging

from os import path
from abc import ABC, abstractmethod

from expert import Expert, Expert_Type

class KF_Model(Expert):
    """The Kalman filter model handles prediction and update of Kalman states.

    Has information about all constant matrices (F, C_w, H and C_v).
    Provides the general prediction and update functionality.

    Attributes:
        F (np.matrix):      The state transition matrix
        C_w (np.matrix):    The prediction covariance matrix
        H (np.matrix):      The measurement matrix
        C_v (np.matrix):    The measurement covariance matrix
        default_state_options (dict): The options for creating a new state taken from the config file
    """

    __metaclass__ = Expert

    def __init__(self, name, model_path, F, C_w, H, C_v, default_state_options):
        """Initialize a new Kalman filter with given matrices."""
        self.F = F
        self.C_w = C_w
        self.H = H
        self.C_v = C_v
        self.default_state_options = default_state_options
        # This variable can be filled in training to perform some of the temporal predictions.
        # This variable will be saved and loaded.
        self.temporal_variable = 0
        # This variable can be filled in training to perform some of the spatial predictions.
        # This variable will be saved and loaded.
        self.spatial_variable = 0
        super().__init__(Expert_Type.KF, name, model_path)

    def predict(self, current_state):
        """Execute the prediction step of the Kalman filter.
            
        Changes the input state to the predicted values.

        Args:
            current_state (np.array): The current state of a particle processed by Kalman filtering
        """
        ## Check matrix and state dimensions
        assert(self.F.shape[0] == current_state.state.shape[0])

        ## Predict next step
        # Predict state x_p = F * x_e
        current_state.state = self.F.dot(current_state.state)
        # Predict C_p = F * C_e * F' + C_w
        current_state.C_p = np.matmul(np.matmul(self.F, current_state.C_e), self.F.T) + self.C_w

    def update(self, current_state, measurement):
        """Execute filtering step of the Kalman filter.

        Args:
            current_state (np.array):   The current state of a particle processed by Kalman filtering
            measurement (list):         The last measurement used for filter step
        """
        ## Check matrix and state dimensions
        assert(self.F.shape[0] == current_state.state.shape[0])
        assert(self.H.shape[0] == len(measurement))
        # First update step needs to be skiped 
        if current_state.first == False:
            measurement = np.array([[measurement[0]], [measurement[1]]])
            # Create a new state
            ## Update (filter) step from last time step
            # Calculate gain K = C_p * H' * inv(C_v + H * C_p * H')
            M1 = self.C_v + np.matmul(np.matmul(self.H, current_state.C_p), self.H.T)
            K = np.matmul(np.matmul(current_state.C_p, self.H.T), M1.I)
            # Update state x_e = x_p + K * (y - H * x_p)
            current_state.state = current_state.state + K.dot(measurement - self.H.dot(current_state.state))
            # Update covariance matrix C_e = C_p - K * H * C_p
            current_state.C_e = current_state.C_p - np.matmul(np.matmul(K, self.H), current_state.C_p)
        else:
            current_state.first = False

    @abstractmethod
    def predict_batch(self, inp):
        """Predict a batch of input data for testing.

        Args:
            inp (tf.Tensor): A batch of input tracks

        Returns
            prediction (np.array): Predicted positions for training instances
        """
        pass

    @abstractmethod
    def predict_batch_separation(self, inp, separation_mask, is_training=False, target=None):
        """Perform separation prediction on a batch of input data.

        Args:
            inp (tf.Tensor):                A batch of input tracks
            separation_mask (tf.Tensor):    Indicating the last time step of the track
            is_training (Boolean):          Only perform default prediction style in training.
            target (tf.Tensor):             The target is needed for training in some cases.

        Returns
            prediction (np.array): Predicted x, y, y_nozzel and dt_nozzle
        """
        pass

    @abstractmethod
    def train_batch(self, inp, target):
        """Train the cv model on a batch of data."""
        pass

    @abstractmethod    
    def train_batch_separation_prediction(self, inp, target, tracking_mask, separation_mask, no_train_mode=False):
        """Train the cv model for separation prediction on a batch of data.

        The cv algorithm will perform tracking and then predict the time and position at the nozzle array.

        Args:
            inp (tf.Tensor):            Batch of track measurements
            target (tf.Tensor):         Batch of track target measurements
            tracking_mask (tf.Tensor):  Batch of tracking masks
            separation_mask (tf.Tensor):Batch of separation masks. Indicates where to start the separation prediction.
            no_train_mode (Boolean):    Option to disable training of spatial and temporal variable

        Returns:
            prediction
            spatial_loss
            temporal_loss
            spatial_mae
            temporal_mae
        """ 
        pass
    
    def test_batch_separation_prediction(self, inp, target, tracking_mask, separation_mask):
        """Call train_batch_separation_prediction in no train mode."""
        prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae = self.train_batch_separation_prediction(inp, target, tracking_mask, separation_mask, True)
        prediction = self.correct_separation_prediction(prediction, np.array(separation_mask))
        return prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae

    def correct_separation_prediction(self, prediction, separation_mask):
        """Correct the uncertainty prediction of the expert with the ENCE calibration.

        Args:
            separation_mask (np.array): Indicates where the separation prediction entries are (end_track)
            prediction (np.array): shape = n_tracks, n_timesteps, 6
                Tracking entries:
                    prediction[i, 0:end_track, 0:2] = [x_pred, y_pred]
                Separation prediction entries:
                    prediction[i, end_track, 2] = y_nozzle_pred    (Predicted y position at nozzle array)
                    prediction[i, end_track, 3] = dt_nozzle_pred   (Predicted time to nozzle array)
                    prediction[i, end_track, 4] = log(var_y)       (Predicted variance of spatial prediction)
                    prediction[i, end_track, 5] = log(var_t)       (Predicted variance of temporal prediction)

        Returns:
            prediction (np.array)
        """
        for track in range(prediction.shape[0]):
            sep_pos = np.where(separation_mask[track] == 1)
            std_y = np.sqrt(np.exp(prediction[track, sep_pos, 4]))
            # spatial correction
            corrected_std_y = self.calibration_separation_regression_var_spatial[0] * std_y + self.calibration_separation_regression_var_spatial[1]
            prediction[track, sep_pos, 4] = np.log(corrected_std_y**2)
            std_t = np.sqrt(np.exp(prediction[track, sep_pos, 5]))
            # temporal correction
            corrected_std_t = self.calibration_separation_regression_var_temporal[0] * std_t + self.calibration_separation_regression_var_temporal[1]
            prediction[track, sep_pos, 5] = np.log(corrected_std_t**2)
        return prediction

    @abstractmethod
    def get_zero_state(self, batch_size):
        """Return a list of dummy states of correct size for the corresponding model.

        Needs to be overwritten by child class.

        Args:
            batch_size (int): The size of the dummy list

        Returns:
            list of dummy states
        """
        pass

    def load_model(self):
        """Load parameters for KF model."""
        if path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.spatial_prediction, self.spatial_variable, self.temporal_prediction, self.temporal_variable = pickle.load(f)
        else:
            logging.warning("Model file for Kalman filter model '{}' does not exist at {}.".format(self.name, self.model_path))
        self.load_calibration()

    def save_model(self):
        """Save parameters for KF model."""
        with open(self.model_path, 'wb') as f:
            pickle.dump([self.spatial_prediction, self.spatial_variable, self.temporal_prediction, self.temporal_variable], f)
        

class KF_State(ABC):
    """The Kalman filter state saves information about the state and covariance matrix of a particle.

    State should include position and velocity.
    Has information about all varying covariance matrices.

    Attributes:
        state (np.array):   The state ((x,y) position, velocity, ...)
        C_p (np.matrix):    The covariance matrix after the prediction step
        C_e (np.matrix):    The covariance matrix after the update step
        first (Boolean):    Is used to skip the first update step because the state is initialized with a measurement.
    """

    def __init__(self, state, C_p, C_e):
        """Initialize a new Kalman filter state (represents one particle)."""
        self.state = state
        self.C_p = C_p
        self.C_e = C_e
        # First update step needs to be skiped 
        self.first = True

    @abstractmethod
    def get_pos(self):
        """Return the x,y position of the state.

        Needs to be overwritten by descenting class.

        Returns:
            [x_pos; y_pos] as a numpy array
        """
        pass

    @abstractmethod
    def get_v(self):
        """Return the velocity of the state.

        Needs to be overwritten by child class.

        Returns:
            [x_velo; y_velo] as a numpy array
        """
        pass