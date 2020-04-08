"""CA Kalman filter model and CA State.

Todo:
    * (Convert np representation to tensor representation for mixture of experts)
"""

import numpy as np

from kf_model import KF_Model, KF_State

class CA_Model(KF_Model):
    """The Constant Acceleration (CA) Kalman filter model.

    Inherites from Kalman filter model.
    """

    __metaclass__ = KF_Model

    def __init__(self, name, dt=0.005, s_w=10E10, s_v=2, default_state_options = {}):
        """Initialize a new model to track particles with the CA Kalman filter.

        Default values are for pixel representation if 1 Pixel = 0.056 mm

        Args:
            name (String):      The name of the expert
            dt=0.005 (double):  The time difference between two measurements
            s_w=10E7 (double):  Power spectral density of particle noise (Highly dependent on particle type)
            s_v=2 (double):     Measurement noise variance (2 is default if input is in pixel)
        """
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
        
        super().__init__(name, F, C_w, H, C_v, default_state_options)

    def train_batch(self, inp, target):
        """Train the cv model on a batch of data."""
        return self.predict_batch(inp)

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