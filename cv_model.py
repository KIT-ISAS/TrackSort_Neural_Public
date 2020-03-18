import numpy as np

from kf_model import KF_Model, KF_State

class CV_Model(KF_Model):
    __metaclass__ = KF_Model

    def __init__(self, dt=0.005, s_w=10E7, s_v=2, default_state_options = {}):
        """
            Initialize a new model to track particles with the CV Kalman filter

            @param dt=0.005:    The time difference between two measurements
            @param s_w=10E7:    Power spectral density of particle noise (Highly dependent on particle type)
            @param s_v=2:       Measurement noise variance (2 is default if input is in pixel)
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

    def get_zero_state(self, batch_size):
        """
            Return a list of dummy CV_States

            @param batch_size: The size of the dummy list

            @return list of dummy CV_States
        """
        dummy_list = []
        for i in range(batch_size):
            dummy_list.append(CV_State([0.0, 0.0], 0))

        return dummy_list

class CV_State(KF_State):
    __metaclass__ = KF_State

    def __init__(self, initial_pos, velocity_guess=1.5E4, velo_var_x=10E6, velo_var_y = 10E8, pos_var=2):
        """
            Initialize a new state of a particle tracked with the constant velocity model

            @param initial_pos:             The initial position of the particle as list [x, y]
            @param velocity_guess=1.5E4:    The initial velocity in x direction of a particle  
            @param pos_var=2:               The initial variance of position (Should be the same as measurement var)
            @param velo_var_x=10E6:         The initial variance of velocity in x direction
            @param velo_var_y=10E8:         The initial variance of velocity in y direction
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
        """
            Returns the x,y position of the state

            @return [x; y] as a numpy array
        """
        return self.state[[0,2],0]

    def get_v(self):
        """
            Returns the velocity of the state

            @return [vx; vy] as a numpy array
        """
        return self.state[[1,3],0]