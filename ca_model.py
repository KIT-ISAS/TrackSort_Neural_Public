"""CA Kalman filter model and CA State.

Todo:
    * (Convert np representation to tensor representation for mixture of experts)
"""

import numpy as np
import logging
from enum import Enum, auto

from kf_model import KF_Model, KF_State


class CA_Temporal_Separation_Type(Enum):
    """Simple enumeration class for temporal separion prediction with the CA model.
    
    Available options are:
        * Default:  The CA motion model
        * LV:       CA model with Limited Velocity - Particles will not be faster than belt speed
    """
    Default = auto()
    LV = auto()

class CA_Spatial_Separation_Type(Enum):
    """Simple enumeration class for spatial separion prediction with the CA model.
    
    Available options are:
        * Default:  The CA motion model
        * DSC:      CA model Disallowing Sign Changes - Particles decellerate until v=0. Then won't change y position.
    """
    Default = auto()
    DSC = auto()

class CA_Model(KF_Model):
    """The Constant Acceleration (CA) Kalman filter model.

    Inherites from Kalman filter model.
    """

    __metaclass__ = KF_Model

    def __init__(self, name, model_path="", x_pred_to = 1550, time_normalization = 22., dt=0.005, s_w=10E10, s_v=2, 
                 temporal_separator = "default", spatial_separator = "default", default_state_options = {}):
        """Initialize a new model to track particles with the CA Kalman filter.

        Default values are for pixel representation if 1 Pixel = 0.056 mm

        Args:
            name (String):                  The name of the expert
            model_path (String):            Path to save/load parameters. Optional for KF because default types do not need training.
            dt=0.005 (double):              The time difference between two measurements
            s_w=10E7 (double):              Power spectral density of particle noise (Highly dependent on particle type)
            s_v=2 (double):                 Measurement noise variance (2 is default if input is in pixel)
            x_pred_to (double):             The x position of the nozzle array (only needed for kf separation prediction)
            time_normalization (double):    Time normalization constant (only needed for kf separation prediction)
            temporal_separator (String):    Temporal separation prediction type ("default", "LV")
            spatial_separator (String):     Spatial separation prediction type ("default", "DSC")
        """
        self.dt = dt        
        self.x_pred_to = x_pred_to
        self.time_normalization = time_normalization
        # Temporal separation prediction type
        if temporal_separator == "default":
            self.temporal_prediction = CA_Temporal_Separation_Type.Default
        elif temporal_separator == "LV":
            self.temporal_prediction = CA_Temporal_Separation_Type.LV
        else:
            logging.warning("Temporal separation prediction type '{}' is unknown. Setting to default.".format(temporal_separator)) 
            self.temporal_prediction = CA_Temporal_Separation_Type.Default
        # Spatial separation prediction type
        if spatial_separator == "default":
            self.spatial_prediction = CA_Spatial_Separation_Type.Default
        elif temporal_separator == "DSC":
            self.spatial_prediction = CA_Spatial_Separation_Type.DSC
        else:
            logging.warning("Spatial separation prediction type '{}' is unknown. Setting to default.".format(spatial_separator)) 
            self.spatial_prediction = CA_Spatial_Separation_Type.Default
        # Transition matrix
        F = np.matrix([[1, dt, dt**2/2, 0, 0, 0],
                       [0, 1, dt, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, dt, dt**2/2],
                       [0, 0, 0, 0, 1, dt],
                       [0, 0, 0, 0, 0, 1]])
        # Prediction covariance matrix
        C_w = s_w * np.matrix([[pow(dt,5)/20, pow(dt,4)/8, pow(dt,3)/6, 0, 0, 0],
                               [pow(dt,4)/8, pow(dt,3)/3, pow(dt,2)/2, 0, 0, 0],
                               [pow(dt,3)/6, pow(dt,2)/2, dt, 0, 0, 0],
                               [0, 0, 0, pow(dt,5)/20, pow(dt,4)/8, pow(dt,3)/6],
                               [0, 0, 0, pow(dt,4)/8, pow(dt,3)/3, pow(dt,2)/2],
                               [0, 0, 0, pow(dt,3)/6, pow(dt,2)/2, dt]])
        # Measurement matrix
        H = np.matrix([[1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0]])
        # Measurement covariance matrix
        C_v = s_v * np.matrix(np.eye(2))
        
        super().__init__(name, model_path, F, C_w, H, C_v, default_state_options)

    def train_batch(self, inp, target):
        """Train the cv model on a batch of data."""
        return self.predict_batch(inp)

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
            prediction (tf.Tensor):     [x_p, y_p, y_nozzle, dt_nozzle]
            spatial_loss (tf.Tensor):   mean((y_nozzle_pred - y_nozzle_target)^2)
            temporal_loss (tf.Tensor):  mean((dt_nozzle_pred - dt_nozzle_target)^2)
            spatial_mae (tf.Tensor):    mean(abs(y_nozzle_pred - y_nozzle_target))
            temporal_mae (tf.Tensor):   mean(abs(dt_nozzle_pred - dt_nozzle_target))
        """ 
        prediction = self.predict_batch_separation(inp=inp, separation_mask=separation_mask, is_training=not no_train_mode)
        target_np = target.numpy()
        separation_mask_np = separation_mask.numpy()
        spatial_loss = np.sum(np.power(prediction[:,:,2]-target_np[:,:,2], 2)*separation_mask_np) / np.sum(separation_mask_np)
        temporal_loss = np.sum(np.power(prediction[:,:,3]-target_np[:,:,3], 2)*separation_mask_np) / np.sum(separation_mask_np)
        spatial_mae = np.sum(np.abs(prediction[:,:,2]-target_np[:,:,2])*separation_mask_np) / np.sum(separation_mask_np)
        temporal_mae = np.sum(np.abs(prediction[:,:,3]-target_np[:,:,3])*separation_mask_np) / np.sum(separation_mask_np)
        return prediction, spatial_loss, temporal_loss, spatial_mae, temporal_mae 

    def predict_batch(self, inp):
        """Predict a batch of data with the cv model."""
        np_inp = inp.numpy()
        predictions = np.zeros(np_inp.shape)
        
        # For each track in batch
        for i in range(np_inp.shape[0]):
            ca_state = CA_State(np_inp[i, 0], **self.default_state_options)
            # For each instance in track
            for j in range(np_inp.shape[1]):
                # Check if track is still alive
                if np.all(np.isclose(np_inp[i, j],[0.0, 0.0])) == False:
                    # update
                    self.update(ca_state, np_inp[i, j])
                    # predict
                    self.predict(ca_state)
                    predictions[i, j, :] = [ca_state.get_pos()[0], ca_state.get_pos()[1]]
                else:
                    break
                    
        return predictions

    def predict_batch_separation(self, inp, separation_mask, is_training=False):
        """Predict a batch of data with the cv model."""
        np_inp = inp.numpy()
        np_separation_mask = separation_mask.numpy()

        n_tracks = np_inp.shape[0]
        track_length = np_inp.shape[1]
        predictions = np.zeros([n_tracks, track_length, 4])
        # For each track in batch
        for i in range(n_tracks):
            ca_state = CA_State(np_inp[i, 0], **self.default_state_options)
            # For each instance in track
            for j in range(track_length):
                # Check if this there are still entries in track
                if np_separation_mask[i, j] == 0:
                    # update
                    self.update(ca_state, np_inp[i, j])
                    # predict
                    self.predict(ca_state)
                    predictions[i, j, :2] = [ca_state.get_pos()[0], ca_state.get_pos()[1]]
                else:
                    if j>= 1:
                        # only perform update step
                        self.update(ca_state, np_inp[i, j])
                        # Make separation prediction
                        a_last = ca_state.get_a()
                        v_last = ca_state.get_v()
                        pos_last = ca_state.get_pos()
                        # In training we only perform the default types
                        temporal_separation_type = CA_Temporal_Separation_Type.Default if is_training else self.temporal_prediction
                        spatial_separation_type = CA_Spatial_Separation_Type.Default if is_training else self.spatial_prediction 
                        # Sanity checks and error management
                        if a_last[0]==0:
                            a_last[0]=0.000000000001
                        sqrt_val = (v_last[0]/a_last[0])**2 - 2*(pos_last[0]-self.x_pred_to)/a_last[0]
                        if sqrt_val < 0:
                            logging.warning("With the predicted velocity of {} and the predicted acceleration of {} the track {} would not reach the nozzle array. Perform cv prediction!".format(v_last[0], a_last[0], i))
                            if v_last[0] > 0:
                                dt_pred = 1/(v_last[0] * self.dt) * (self.x_pred_to-pos_last[0])
                                y_pred = pos_last[1] + dt_pred * v_last[1] * self.dt
                            else:
                                logging.warning("The predicted velocity in x direction was {} <= 0 in track {} using the CV KF model.".format(v_last[0], i))
                                y_pred = pos_last[1]
                                dt_pred = 11 # This is a fairly close value. Please investigate the issue!   
                            break 
                        # First perform temporal predicion
                        if temporal_separation_type == CA_Temporal_Separation_Type.Default:
                            dt_pred = 1/self.dt * (- v_last[0]/a_last[0] + np.sign(a_last[0]) * np.sqrt(sqrt_val))
                        elif temporal_separation_type == CA_Temporal_Separation_Type.LV:
                            logging.error("LV prediction not implemented yet.")
                            dt_pred = 11
                        # Then perform spatial prediction
                        if spatial_separation_type == CA_Spatial_Separation_Type.Default:
                            y_pred = pos_last[1] + dt_pred * v_last[1] * self.dt + 1/2 * dt_pred**2 * self.dt**2 * a_last[1]
                        elif spatial_separation_type == CA_Spatial_Separation_Type.DSC:
                            logging.error("DSC prediction not implemented yet.")
                            y_pred = pos_last[1]
                                       
                        predictions[i, j, 2] = y_pred
                        predictions[i, j, 3] = dt_pred/self.time_normalization
                    else:
                        logging.warning("Track out of measurements at first time step.")
                    break

        return predictions

    def get_zero_state(self, batch_size):
        """Return a list of dummy CV_States."""
        dummy_list = []
        for i in range(batch_size):
            dummy_list.append(CA_State([0.0, 0.0], 0))

        return dummy_list


class CA_State(KF_State):
    """The Constant Acceleration Kalman filter state saves information about the state and covariance matrix of a particle.

    State vector is: [x_pos,
                      x_velo,
                      x_accel,
                      y_pos,
                      y_velo,
                      y_accel]
    """

    __metaclass__ = KF_State

    def __init__(self, initial_pos, velocity_guess=1.5E4, velo_var_x=10E6, velo_var_y = 10E8, pos_var=2, accel_guess = 0.0, accel_var_x = 10E8, accel_var_y = 10E9):
        """Initialize a new state of a particle tracked with the constant acceleration model.

        Default values are for pixel representation if 1 Pixel = 0.056 mm

        Args:
            initial_pos (list):      The initial position of the particle as list [x, y]
            velocity_guess=1.5E4:    The initial velocity in x direction of a particle  
            accel_guess=0            The initial acceleration in x direction of a particle   
            pos_var=2:               The initial variance of position (Should be the same as measurement var)
            velo_var_x=10E6:         The initial variance of velocity in x direction
            velo_var_y=10E8:         The initial variance of velocity in y direction
            accel_var_x = 10E8:      The initial variance of acceleration in x direction
            accel_var_y = 10E9:      The initial variance of acceleration in y direction
        """
        # state=[x, v_x, a_x, y, v_y, a_y]'
        state = np.array([[initial_pos[0]], [velocity_guess], [accel_guess], [initial_pos[1]], [0], [0]])
        # Prediction covariance matrix
        C_p = np.matrix([[pos_var, 0, 0, 0, 0, 0],
                         [0, velo_var_x, 0, 0, 0, 0],
                         [0, 0, accel_var_x, 0, 0, 0],
                         [0, 0, 0, pos_var, 0, 0],
                         [0, 0, 0, 0, velo_var_y, 0],
                         [0, 0, 0, 0, 0, accel_var_y]])
        # Estimate covariance matrix
        C_e = np.matrix([[pos_var, 0, 0, 0, 0, 0],
                         [0, velo_var_x, 0, 0, 0, 0],
                         [0, 0, accel_var_x, 0, 0, 0],
                         [0, 0, 0, pos_var, 0, 0],
                         [0, 0, 0, 0, velo_var_y, 0],
                         [0, 0, 0, 0, 0, accel_var_y]])

        super().__init__(state, C_p, C_e)

    def get_pos(self):
        """Return the x,y position of the state."""
        return self.state[[0,3],0]

    def get_v(self):
        """Return the velocity of the state."""
        return self.state[[1,4],0]

    def get_a(self):
        """Return the acceleration of the state."""
        return self.state[[2,5],0]