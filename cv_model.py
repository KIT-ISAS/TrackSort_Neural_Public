"""CV Kalman filter model and CV State.

Todo:
    * Convert np representation to tensor representation for mixture of experts
"""

import numpy as np

from kf_model import KF_Model, KF_State

class CV_Model(KF_Model):
    """The Constant Velocity (CV) Kalman filter model.

    Inherites from Kalman filter model.
    """

    __metaclass__ = KF_Model

    def __init__(self, dt=0.005, s_w=10E7, s_v=2, default_state_options = {}):
        """Initialize a new model to track particles with the CV Kalman filter.

        Default values are for pixel representation if 1 Pixel = 0.056 mm

        Args:
            dt=0.005:    The time difference between two measurements
            s_w=10E7:    Power spectral density of particle noise (Highly dependent on particle type)
            s_v=2:       Measurement noise variance (2 is default if input is in pixel)
        """
        # Transition matrix
        F = np.matrix([[1, dt, 0, 0],
                           [0,  1, 0, 0],
                           [0,  0, 1, dt],
                           [0,  0, 0, 1]])
        # Prediction covariance matrix
        C_w = s_w * np.matrix([[pow(dt,3)/3, pow(dt,2)/2,           0,           0],
                                [pow(dt,2)/2,          dt,           0,           0],
                                [          0,           0, pow(dt,3)/3, pow(dt,2)/2],
                                [          0,           0, pow(dt,2)/2,          dt]])
        # Measurement matrix
        H = np.matrix([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        # Measurement covariance matrix
        C_v = s_v * np.matrix(np.eye(2))
        
        super().__init__(F, C_w, H, C_v, default_state_options)

    def train_batch(self, inp, target):
        """Train the cv model on a batch of data."""
        np_inp = inp.numpy()
        np_target = target.numpy()
        predictions = []
        all_mse = []
        mse_sum = 0
        all_mae = []
        mae_sum = 0
        c = 0
        for i in range(np_inp.shape[0]):
            cv_state = CV_State(np_inp[i, 0], **self.default_state_options)
            mse_track = []
            mae_track = []
            track_prediction = []
            for j in range(np_inp.shape[1]):
                if np.all(np.isclose(np_target[i, j],[0.0, 0.0])) == False:
                    # update
                    self.update(cv_state, np_inp[i, j])
                    # predict
                    self.predict(cv_state)
                    pos = cv_state.get_pos()
                    track_prediction.append(pos)
                    current_mse = (pos[0]-np_target[i, j, 0])**2 + (pos[1]-np_target[i, j, 1])**2
                    mse_track.append(current_mse.item(0))
                    mse_sum = mse_sum + current_mse.item(0)
                    current_mae = np.abs(pos[0]-np_target[i, j, 0]) + np.abs(pos[1]-np_target[i, j, 1])
                    mae_track.append(current_mae.item(0)) 
                    mae_sum = mae_sum + current_mae.item(0)  
                    c = c+1
            all_mse.append(mse_track)
            all_mae.append(mae_track)
            predictions.append(np.array(track_prediction))
        mse = mse_sum/c
        mae = mae_sum/c
        return np.array(predictions)

    def get_zero_state(self, batch_size):
        """Return a list of dummy CV_States."""
        dummy_list = []
        for i in range(batch_size):
            dummy_list.append(CV_State([0.0, 0.0], 0))

        return dummy_list

class CV_State(KF_State):
    """The Constant Velocity Kalman filter state saves information about the state and covariance matrix of a particle.

    State vector is: [x_pos,
                      x_velo,
                      y_pos,
                      y_velo]
    """

    __metaclass__ = KF_State

    def __init__(self, initial_pos, velocity_guess=1.5E4, velo_var_x=10E6, velo_var_y = 10E8, pos_var=2):
        """Initialize a new state of a particle tracked with the constant velocity model.

        Default values are for pixel representation if 1 Pixel = 0.056 mm

        Args:
            initial_pos (list):      The initial position of the particle as list [x, y]
            velocity_guess=1.5E4:    he initial velocity in x direction of a particle  
            pos_var=2:               The initial variance of position (Should be the same as measurement var)
            velo_var_x=10E6:         The initial variance of velocity in x direction
            velo_var_y=10E8:         The initial variance of velocity in y direction
        """
        # state=[x, v_x, y, v_y]'
        state = np.array([[initial_pos[0]], [velocity_guess], [initial_pos[1]], [0]])
        # Prediction covariance matrix
        C_p = np.matrix([[pos_var, 0, 0, 0],
                             [0, velo_var_x, 0, 0],
                             [0, 0, pos_var, 0],
                             [0, 0, 0, velo_var_y]])
        # Estimate covariance matrix
        C_e = np.matrix([[pos_var, 0, 0, 0],
                             [0, velo_var_x, 0, 0],
                             [0, 0, pos_var, 0],
                             [0, 0, 0, velo_var_y]])

        super().__init__(state, C_p, C_e)

    def get_pos(self):
        """Return the x,y position of the state."""
        return self.state[[0,2],0]

    def get_v(self):
        """Return the velocity of the state."""
        return self.state[[1,3],0]