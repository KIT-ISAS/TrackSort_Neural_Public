import numpy as np

class CV_Model(object):
    

    def __init__(self, initial_pos, dt=0.005, s_w=1, velocity_guess=30):
        """
            Initialize a new particle tracked with the cv model

            @param initial_pos:       The initial position of the particle as list [x, y]
            @param dt=0.005:          The time difference between two measurements
            @param s_w=1:             Power spectral density of particle noise
            @param velocity_guess=30: The initial velocity in x direction of a particle     
        """
        # Transition matrix
        self.F = np.array([[1, dt, 0, 0],
                           [0,  1, 0, 0],
                           [0,  0, 1, dt],
                           [0,  0, 0, 1]])
        # Prediction covariance matrix
        self.C_w = np.array(s_w*[[pow(dt,3)/3, pow(dt,2)/2,           0,           0],
                                [pow(dt,2)/2,          dt,           0,           0],
                                [          0,           0, pow(dt,3)/3, pow(dt,2)/2],
                                [          0,           0, pow(dt,2)/2,          dt]])
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        # Measurement covariance matrix
        self.C_v = np.eye(2) * 2 * velocity_guess/30
        # Changing state of the particle
        self.state = CV_State(initial_pos, velocity_guess)
        
    def predict(self, measurement):
        """
            Execute prediction and filtering step of Kalman filter

            @param measurement: The last measurement used for filter step
            @param state:       The current state containing the uncertainty 

            @return     The updated state
        """
        ## Update (filter) step from last time step
        # Calculate gain K
        K = self.state.C_p * self.H * np.invert(self.C_v + self.H * self.state.C_p * np.transpose(self.H))
        # Update state
        self.state.state = self.state.state + K * (measurement - self.H * self.state.state)
        # Update covariance matrix
        self.state.C_e = self.state.C_p - K * self.H * self.state.C_p

        ## Predict next step
        self.state.state = self.F * self.state.state
        self.state.C_p = self.F * self.state.C_p * self.F + self.C_w

    def get_pos(self):
        """
            Returns the x,y position of the state

            @return [x; y] as a numpy array
        """
        return self.state.state[[0,2],0]

class CV_State(object):
    def __init__(self, initial_pos, velocity_guess=30):
        """
            Initialize a new state of a particle tracked with the constant velocity model

            @param initial_pos:       The initial position of the particle as list [x, y]
            @param velocity_guess=30: The initial velocity in x direction of a particle     
        """

        # state=[x, v_x, y, v_y]'
        self.state = np.transpose(np.array([initial_pos[0], velocity_guess, initial_pos[1], 0]))
        # Prediction covariance matrix
        scale = velocity_guess/30
        pos_var = 2*scale
        velo_var = 5*scale
        self.C_p = np.array([[pos_var, 0, 0, 0],
                             [0, pos_var, 0, 0],
                             [0, 0, velo_var, 0],
                             [0, 0, 0, velo_var]])
        # Estimate covariance matrix
        self.C_e = np.zeros([4,4])
