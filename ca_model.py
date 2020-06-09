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
        * CA:  The CA motion model
        * CVBC:     The bias corrected CV motion model
        * IA:     Identically Acceleration model. The median acceleration is learned on training data. CA model with the median acc is applied.
        * LV:       CA model with Limited Velocity - Particles will not be faster than belt speed
    """
    CA = auto()
    CVBC = auto()
    IA = auto()
    LV = auto()

class CA_Spatial_Separation_Type(Enum):
    """Simple enumeration class for spatial separion prediction with the CA model.
    
    Available options are:
        * CA:       The CA motion model
        * CV:       The CV motion model
        * Ratio:    The ratio of speed change is learned on training data and applied with CA model.
        * DSC:      CA model Disallowing Sign Changes - Particles decellerate until v=0. Then won't change y position.
    """
    CA = auto()
    CV = auto()
    Ratio = auto()
    DSC = auto()

class CA_Model(KF_Model):
    """The Constant Acceleration (CA) Kalman filter model.

    Inherites from Kalman filter model.
    """

    __metaclass__ = KF_Model

    def __init__(self, name, model_path="", x_pred_to = 1550, time_normalization = 22., dt=0.005, s_w=10E10, s_v=2, 
                 temporal_separator = "CA", spatial_separator = "CA", belt_velocity = 1.5, default_state_options = {}):
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
            temporal_separator (String):    Temporal separation prediction type ("CA", "CVBC", "IA", "LV")
            spatial_separator (String):     Spatial separation prediction type ("CA", "CV", "ratio", "DSC")
            belt_velocity (double):         The velocity of the belt. Only needed for temporal prediction type limited velocity "LV".
        """
        self.dt = dt        
        self.x_pred_to = x_pred_to
        self.time_normalization = time_normalization
        self.belt_velocity = belt_velocity
        # Temporal separation prediction type
        if temporal_separator == "CA":
            self.temporal_prediction = CA_Temporal_Separation_Type.CA
        elif temporal_separator == "CVBC":
            self.temporal_prediction = CA_Temporal_Separation_Type.CVBC
        elif temporal_separator == "IA":
            self.temporal_prediction = CA_Temporal_Separation_Type.IA
        elif temporal_separator == "LV":
            self.temporal_prediction = CA_Temporal_Separation_Type.LV
        else:
            logging.warning("Temporal separation prediction type '{}' is unknown. Setting to default = CA.".format(temporal_separator)) 
            self.temporal_prediction = CA_Temporal_Separation_Type.CA
        # Spatial separation prediction type
        if spatial_separator == "CA":
            self.spatial_prediction = CA_Spatial_Separation_Type.CA
        elif spatial_separator == "CV":
            self.spatial_prediction = CA_Spatial_Separation_Type.CV
        elif spatial_separator == "ratio":
            self.spatial_prediction = CA_Spatial_Separation_Type.Ratio
        elif spatial_separator == "DSC":
            self.spatial_prediction = CA_Spatial_Separation_Type.DSC
        else:
            logging.warning("Spatial separation prediction type '{}' is unknown. Setting to default = CA.".format(spatial_separator)) 
            self.spatial_prediction = CA_Spatial_Separation_Type.CA
        
        self.temporal_training_list = []
        self.spatial_training_list = []
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
        prediction = self.predict_batch_separation(inp=inp, separation_mask=separation_mask, is_training=not no_train_mode, target = target)
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

    def predict_batch_separation(self, inp, separation_mask, is_training=False, target=None):
        """Predict a batch of data with the cv model."""
        np_inp = inp.numpy()
        np_separation_mask = separation_mask.numpy()
        if target is not None:
            target_np = target.numpy()
        else:
            target_np = None
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
                        # Sanity checks and error management
                        if a_last[0,0]==0:
                            a_last[0,0]=1E-20
                        sqrt_val = (v_last[0,0]/a_last[0,0])**2 - 2*(pos_last[0,0]-self.x_pred_to)/a_last[0,0]
                        if sqrt_val < 0:
                            logging.warning("With the predicted velocity of {} and the predicted acceleration of {} the track {} would not reach the nozzle array. Perform cv prediction!".format(v_last[0], a_last[0], i))
                            if v_last[0,0] > 0:
                                dt_pred = 1/(v_last[0,0] * self.dt) * (self.x_pred_to-pos_last[0,0])
                                y_pred = pos_last[1,0] + dt_pred * v_last[1,0] * self.dt
                            else:
                                logging.warning("The predicted velocity in x direction was {} <= 0 in track {} using the CV KF model.".format(v_last[0], i))
                                y_pred = pos_last[1,0]
                                dt_pred = 11 # This is a fairly close value. Please investigate the issue!   
                            break 
                        ## First perform temporal predicion
                        target_time = target_np[i, j, 3]*self.time_normalization
                        if self.temporal_prediction == CA_Temporal_Separation_Type.CA:
                            dt_pred = 1/self.dt * (- v_last[0,0]/a_last[0,0] + np.sign(a_last[0,0]) * np.sqrt(sqrt_val))
                        elif self.temporal_prediction == CA_Temporal_Separation_Type.CVBC:
                            dt_pred = 1/(v_last[0,0] * self.dt) * (self.x_pred_to-pos_last[0,0])
                            if is_training:
                                self.temporal_training_list.append(dt_pred - target_time)
                            else:
                                # Adjust prediction with bias
                                dt_pred -= self.temporal_variable
                        elif self.temporal_prediction == CA_Temporal_Separation_Type.IA:
                            dt_pred = 1/(v_last[0,0] * self.dt) * (self.x_pred_to-pos_last[0,0])
                            if is_training:
                                # Find optimal acceleration value
                                a_x_opt = 2*(self.x_pred_to-pos_last[0,0]-v_last[0,0] * self.dt * target_time)/(target_time * self.dt)**2
                                self.temporal_training_list.append(a_x_opt)
                            else:
                                # Perform CA prediction with a_opt instead of a_last
                                a_best = self.temporal_variable
                                sqrt_val = (v_last[0,0]/a_best)**2 - 2*(pos_last[0,0]-self.x_pred_to)/a_best
                                if sqrt_val >= 0:
                                    dt_pred = 1/self.dt * (- v_last[0,0]/a_best + np.sign(a_best) * np.sqrt(sqrt_val))
                                else:
                                    logging.warning("Can not perform IA prediction with last x velocity of {} and best acceleration {}.".format(v_last[0], a_best))
                        elif self.temporal_prediction == CA_Temporal_Separation_Type.LV:
                            dt_pred = 1/self.dt * (- v_last[0,0]/a_last[0,0] + np.sign(a_last[0,0]) * np.sqrt(sqrt_val))
                            if ~is_training:
                                t_max_vel = 1/self.dt * (self.belt_velocity-v_last[0,0])/a_last[0,0]
                                # If the paticle x velocity would exceed the belt velocity before the nozzle array
                                if t_max_vel < dt_pred and t_max_vel>0:
                                    # Distance to exceeding point
                                    x_ca = pos_last[0,0] + v_last[0,0]  * self.dt * t_max_vel + 1/2 * a_last[0,0] * (self.dt * t_max_vel)**2
                                    # CV with v_x = v_belt afterwards
                                    dt_pred = t_max_vel + 1/self.dt * (self.x_pred_to - x_ca)/self.belt_velocity
                                
                        ## Then perform spatial prediction
                        if a_last[1,0]==0:
                            a_last[1,0]=1E-20
                        if v_last[1,0]==0:
                            v_last[1,0]=1E-20
                        if self.spatial_prediction == CA_Spatial_Separation_Type.CA:
                            y_pred = pos_last[1,0] + v_last[1,0] * dt_pred * self.dt + 1/2 * (dt_pred * self.dt)**2 * a_last[1,0]
                        elif self.spatial_prediction == CA_Spatial_Separation_Type.DSC:
                            y_pred = pos_last[1,0] + v_last[1,0] * dt_pred * self.dt + 1/2 * (dt_pred * self.dt)**2 * a_last[1,0]
                            if ~is_training:
                                t_sign_change = 1/self.dt * (0-v_last[1,0])/a_last[1,0]
                                # If the paticle y velocity would hit a sign change before the nozzle array
                                if t_sign_change < dt_pred and t_sign_change > 0:
                                    # Y position is fix after t sign change
                                    y_pred = pos_last[1,0] + v_last[1,0] * t_sign_change * self.dt + 1/2 * (t_sign_change * self.dt)**2 * a_last[1,0]
                        elif self.spatial_prediction == CA_Spatial_Separation_Type.CV:
                            y_pred = pos_last[1,0] + dt_pred * v_last[1,0] * self.dt
                        elif self.spatial_prediction == CA_Spatial_Separation_Type.Ratio:
                            y_pred = pos_last[1,0] + dt_pred * v_last[1,0] * self.dt
                            if is_training:
                                if j>=2:
                                    v_nozzle_target = target_np[i, j, 4]/self.dt 
                                    r_i = v_nozzle_target/v_last[1,0]
                                    self.spatial_training_list.append(r_i)
                            else:
                                r = self.spatial_variable
                                a_ratio = -(1-r)*v_last[1,0]/(dt_pred*self.dt)
                                y_pred = pos_last[1,0] + v_last[1,0] * dt_pred * self.dt + 1/2 * (dt_pred * self.dt)**2 * a_ratio

                        predictions[i, j, 2] = y_pred
                        predictions[i, j, 3] = dt_pred/self.time_normalization
                    else:
                        logging.warning("Track out of measurements at first time step.")
                    break

        if is_training:
            if len(self.temporal_training_list)>0:
                self.temporal_variable = np.nanmedian(self.temporal_training_list)
            if len(self.spatial_training_list)>0:
                self.spatial_variable = np.nanmedian(self.spatial_training_list)
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