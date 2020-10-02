"""DQ model.

The DQ model does not track particles and is only for separation prediction.
The velocity and acceleration are built with the last two or three measurements.
Florian Pfaff used this model in his PhD thesis for the separation prediction.
If you use this model with on simulation data without additional noise, you will get his results.

Change log (Please insert your name here if you worked on this file)
    * Created by: Jakob Thumm (jakob.thumm@student.kit.edu)
    * Jakob Thumm 2.10.2020:    Completed documentation.
"""

import numpy as np
import logging
from enum import Enum, auto

from kf_model import KF_Model, KF_State


class DQ_Temporal_Separation_Type(Enum):
    """Simple enumeration class for temporal separion prediction with the DQ model.
    
    Available options are:
        * CA:       The CA motion model
        * CVBC:     The bias corrected CV motion model
        * IA:       Identically Acceleration model. The median acceleration is learned on training data. CA model with the median acc is applied.
        * LV:       CA model with Limited Velocity - Particles will not be faster than belt speed
    """
    CA = auto()
    CVBC = auto()
    IA = auto()
    LV = auto()

class DQ_Spatial_Separation_Type(Enum):
    """Simple enumeration class for spatial separion prediction with the DQ model.
    
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

class DQ_Model(KF_Model):
    """Difference quotientmodel.

    Inherites from Kalman filter model. (Yeah, does not really fit in but who cares.)
    """

    __metaclass__ = KF_Model

    def __init__(self, name, model_path="", x_pred_to = 1550, time_normalization = 22., dt=0.005, 
                 temporal_separator = "default", spatial_separator = "default", belt_velocity = 1.5):
        """Initialize a new model to track particles with the DQ model.

        Args:
            name (String):                  The name of the expert
            model_path (String):            Path to save/load parameters. Optional for KF because default types do not need training.
            dt=0.005 (double):              The time difference between two measurements
            x_pred_to (double):             The x position of the nozzle array (only needed for kf separation prediction)
            time_normalization (double):    Time normalization constant (only needed for kf separation prediction)
            temporal_separator (String):    Temporal separation prediction type ("default", "LV")
            spatial_separator (String):     Spatial separation prediction type ("default", "DSC")
            belt_velocity (double):         The velocity of the belt. Only needed for temporal prediction type limited velocity "LV".
        """
        self.dt = dt        
        self.x_pred_to = x_pred_to
        self.time_normalization = time_normalization
        self.belt_velocity = belt_velocity
        # Temporal separation prediction type
        if temporal_separator == "CA":
            self.temporal_prediction = DQ_Temporal_Separation_Type.CA
        elif temporal_separator == "CVBC":
            self.temporal_prediction = DQ_Temporal_Separation_Type.CVBC
        elif temporal_separator == "IA":
            self.temporal_prediction = DQ_Temporal_Separation_Type.IA
        elif temporal_separator == "LV":
            self.temporal_prediction = DQ_Temporal_Separation_Type.LV
        else:
            logging.warning("Temporal separation prediction type '{}' is unknown. Setting to default.".format(temporal_separator)) 
            self.temporal_prediction = DQ_Temporal_Separation_Type.CA
        # Spatial separation prediction type
        if spatial_separator == "CA":
            self.spatial_prediction = DQ_Spatial_Separation_Type.CA
        elif spatial_separator == "CV":
            self.spatial_prediction = DQ_Spatial_Separation_Type.CV
        elif spatial_separator == "ratio":
            self.spatial_prediction = DQ_Spatial_Separation_Type.Ratio
        elif spatial_separator == "DSC":
            self.spatial_prediction = DQ_Spatial_Separation_Type.DSC
        else:
            logging.warning("Spatial separation prediction type '{}' is unknown. Setting to default.".format(spatial_separator)) 
            self.spatial_prediction = DQ_Spatial_Separation_Type.CA
        
        self.temporal_training_list = []
        self.spatial_training_list = []
        ## F and C_w are only created for the super() call
        # Transition matrix
        F = np.matrix([[1, dt, dt**2/2, 0, 0, 0],
                       [0, 1, dt, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, dt, dt**2/2],
                       [0, 0, 0, 0, 1, dt],
                       [0, 0, 0, 0, 0, 1]])
        # Prediction covariance matrix
        C_w = 1 * np.matrix([[pow(dt,5)/20, pow(dt,4)/8, pow(dt,3)/6, 0, 0, 0],
                               [pow(dt,4)/8, pow(dt,3)/3, pow(dt,2)/2, 0, 0, 0],
                               [pow(dt,3)/6, pow(dt,2)/2, dt, 0, 0, 0],
                               [0, 0, 0, pow(dt,5)/20, pow(dt,4)/8, pow(dt,3)/6],
                               [0, 0, 0, pow(dt,4)/8, pow(dt,3)/3, pow(dt,2)/2],
                               [0, 0, 0, pow(dt,3)/6, pow(dt,2)/2, dt]])
        # Measurement matrix
        H = np.matrix([[1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0]])
        # Measurement covariance matrix
        C_v = 1 * np.matrix(np.eye(2))
        
        super().__init__(name, model_path, F, C_w, H, C_v, {})

    def train_batch(self, inp, target):
        """Train the cv model on a batch of data."""
        logging.error("The DQ model is only for separation prediction. Sooorry.")
        pass

    def train_batch_separation_prediction(self, inp, target, tracking_mask, separation_mask, no_train_mode=False):
        """Train the cv model for separation prediction on a batch of data.

        The cv algorithm will perform tracking and then predict the time and position at the nozzle array.

        Args:
            inp (tf.Tensor):            Batch of track measurements, [x, y], shape = [n_tracks, track_length, 2]
            target (tf.Tensor):         Batch of track target measurements, [x, y, y_nozzle, dt_nozzle, vx_nozzle], shape = [n_tracks, track_length, 5]
            tracking_mask (tf.Tensor):  Batch of tracking masks, shape = [n_tracks, track_length]
            separation_mask (tf.Tensor):Batch of separation masks. Indicates where to start the separation prediction, shape = [n_tracks, track_length]
            no_train_mode (Boolean):    Option to disable training of spatial and temporal variable

        Returns:
            prediction (tf.Tensor):     [x_p, y_p, y_nozzle, dt_nozzle], shape = [n_tracks, track_length, 4]
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
        """Not valid for DQ."""
        logging.error("The DQ model is only for separation prediction. Sooorry.")
        pass

    def predict_batch_separation(self, inp, separation_mask, is_training=False, target=None):
        """Predict a batch of data for separation prediction with the DQ model.

        Training is enabled by passing is_training=True and giving a target array

        Args:
            inp (tf.Tensor):            Batch of track measurements, [x, y], shape = [n_tracks, track_length, 2]
            target (tf.Tensor):         Batch of track target measurements, [x, y, y_nozzle, dt_nozzle, v_nozzle], shape = [n_tracks, track_length, 5]
            separation_mask (tf.Tensor):Batch of separation masks. Indicates where to start the separation prediction, shape = [n_tracks, track_length]   
            is_training (Boolean):      Activate training 
        
        Returns:
            prediction (np.array): shape = n_tracks, n_timesteps, 4
                Tracking entries:
                    prediction[i, 0:end_track, 0:2] = [0, 0]
                Separation prediction entries:
                    prediction[i, end_track, 2] = y_nozzle_pred   (Predicted y position at nozzle array)
                    prediction[i, end_track, 3] = t_nozzle_pred   (Predicted time to nozzle array)
        """
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
            # For each instance in track
            for j in range(track_length):
                # Check if this there are still entries in track
                if np_separation_mask[i, j] == 1:
                    if j>= 2:
                        # get the last 3 measurements
                        pos_0 = np_inp[i, j-2, 0:2]
                        pos_1 = np_inp[i, j-1, 0:2]
                        pos_2 = np_inp[i, j, 0:2]
                        # Make separation prediction
                        a_last = ((pos_2-pos_1)-(pos_1-pos_0))/(self.dt**2)
                        v_last = (pos_2-pos_1)/self.dt
                        pos_last = pos_2
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
                        ## First perform temporal predicion
                        target_time = target_np[i, j, 3]*self.time_normalization
                        if self.temporal_prediction == DQ_Temporal_Separation_Type.CA:
                            roots = np.roots([1/2*a_last[0], v_last[0], pos_last[0]-self.x_pred_to])
                            dt_pred=np.min(roots[roots>=0])
                            #dt_pred = 1/self.dt * (- v_last[0]/a_last[0] + np.sign(a_last[0]) * np.sqrt(sqrt_val))
                            stop = 0
                        elif self.temporal_prediction == DQ_Temporal_Separation_Type.CVBC:
                            dt_pred = 1/(v_last[0] * self.dt) * (self.x_pred_to-pos_last[0])
                            if is_training:
                                self.temporal_training_list.append(dt_pred - target_time)
                            else:
                                # Adjust prediction with bias
                                dt_pred -= self.temporal_variable
                        elif self.temporal_prediction == DQ_Temporal_Separation_Type.IA:
                            dt_pred = 1/(v_last[0] * self.dt) * (self.x_pred_to-pos_last[0])
                            if is_training:
                                # Find optimal acceleration value
                                a_x_opt = 2*(self.x_pred_to-pos_last[0]-v_last[0] * self.dt * target_time)/(target_time * self.dt)**2
                                self.temporal_training_list.append(a_x_opt)
                            else:
                                # Perform DQ prediction with a_opt instead of a_last
                                a_best = self.temporal_variable
                                sqrt_val = (v_last[0]/a_best)**2 - 2*(pos_last[0]-self.x_pred_to)/a_best
                                if sqrt_val >= 0:
                                    dt_pred = 1/self.dt * (- v_last[0]/a_best + np.sign(a_best) * np.sqrt(sqrt_val))
                                else:
                                    logging.warning("Can not perform IA prediction with last x velocity of {} and best acceleration {}.".format(v_last[0], a_best))
                        elif self.temporal_prediction == DQ_Temporal_Separation_Type.LV:
                            dt_pred = 1/self.dt * (- v_last[0]/a_last[0] + np.sign(a_last[0]) * np.sqrt(sqrt_val))
                            if ~is_training:
                                t_max_vel = 1/self.dt * (self.belt_velocity-v_last[0])/a_last[0]
                                # If the paticle x velocity would exceed the belt velocity before the nozzle array
                                if t_max_vel < dt_pred and t_max_vel>0:
                                    # Distance to exceeding point
                                    x_ca = pos_last[0] + v_last[0]  * self.dt * t_max_vel + 1/2 * a_last[0] * (self.dt * t_max_vel)**2
                                    # CV with v_x = v_belt afterwards
                                    dt_pred = t_max_vel + 1/self.dt * (self.x_pred_to - x_ca)/self.belt_velocity
                                
                        ## Then perform spatial prediction
                        if a_last[1]==0:
                            a_last[1]=1E-20
                        if v_last[1]==0:
                            v_last[1]=1E-20
                        if self.spatial_prediction == DQ_Spatial_Separation_Type.CA:
                            y_pred = pos_last[1] + v_last[1] * dt_pred * self.dt + 1/2 * (dt_pred * self.dt)**2 * a_last[1]
                        elif self.spatial_prediction == DQ_Spatial_Separation_Type.DSC:
                            y_pred = pos_last[1] + v_last[1] * dt_pred * self.dt + 1/2 * (dt_pred * self.dt)**2 * a_last[1]
                            if ~is_training:
                                t_sign_change = 1/self.dt * (0-v_last[1])/a_last[1]
                                # If the paticle y velocity would hit a sign change before the nozzle array
                                if t_sign_change < dt_pred:
                                    # Y position is fix after t sign change
                                    y_pred = pos_last[1] + v_last[1] * t_sign_change * self.dt + 1/2 * (t_sign_change * self.dt)**2 * a_last[1]
                        elif self.spatial_prediction == DQ_Spatial_Separation_Type.CV:
                            y_pred = pos_last[1] + dt_pred * v_last[1] * self.dt
                        elif self.spatial_prediction == DQ_Spatial_Separation_Type.Ratio:
                            y_pred = pos_last[1] + dt_pred * v_last[1] * self.dt
                            if is_training:
                                if j>=2:
                                    v_nozzle_target = target_np[i, j, 4]/self.dt 
                                    r_i = v_nozzle_target/v_last[1]
                                    self.spatial_training_list.append(r_i)
                            else:
                                r = self.spatial_variable
                                a_ratio = -(1-r)*v_last[1]/(dt_pred*self.dt)
                                y_pred = pos_last[1] + v_last[1] * dt_pred * self.dt + 1/2 * (dt_pred * self.dt)**2 * a_ratio

                        predictions[i, j, 2] = y_pred
                        predictions[i, j, 3] = dt_pred/self.time_normalization
                    else:
                        logging.warning("Track out of measurements at the second step.")
                    break

        if is_training:
            if len(self.temporal_training_list)>0:
                self.temporal_variable = np.nanmedian(self.temporal_training_list)
            if len(self.spatial_training_list)>0:
                self.spatial_variable = np.nanmedian(self.spatial_training_list)
        return predictions

    def get_zero_state(self, batch_size):
        """Not valid for DQ."""
        logging.error("The DQ model is only for separation prediction. Sooorry.")
        pass


