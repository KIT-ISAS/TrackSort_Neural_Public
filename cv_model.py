import numpy as np

class CV_Model(object):
    

    def __init__(self, dt=0.005, s_w=100000, s_v=1):
        """
            Initialize a new particle tracked with the cv model

            @param dt=0.005:    The time difference between two measurements
            @param s_w=1000:    Power spectral density of particle noise
            @param s_v=1:       Measurement noise  
        """
        # Transition matrix
        self.F = np.matrix([[1, dt, 0, 0],
                           [0,  1, 0, 0],
                           [0,  0, 1, dt],
                           [0,  0, 0, 1]])
        # Prediction covariance matrix
        self.C_w = s_w * np.matrix([[pow(dt,3)/3, pow(dt,2)/2,           0,           0],
                                [pow(dt,2)/2,          dt,           0,           0],
                                [          0,           0, pow(dt,3)/3, pow(dt,2)/2],
                                [          0,           0, pow(dt,2)/2,          dt]])
        # Measurement matrix
        self.H = np.matrix([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        # Measurement covariance matrix
        self.C_v = s_v * np.matrix(np.eye(2)) #TODO: Find good values
        # Changing state of the particle
        #self.state = CV_State(initial_pos, velocity_guess)
        
    def predict(self, current_state):
        """
            Execute prediction step of Kalman filter

            @param current_state: The current state of a particle processed by Kalman filtering
            @param state:         The current state containing the uncertainty 

            @return     The predicted state
        """
        ## Predict next step
        # Predict state x_p = F * x_e
        current_state.state = self.F.dot(current_state.state)
        # Predict C_p = F * C_e * F' + C_w
        current_state.C_p = np.matmul(np.matmul(self.F, current_state.C_e), self.F.T) + self.C_w
        return current_state
    
    def update(self, current_state, measurement):
        """
            Execute filtering step of Kalman filter

            @param current_state: The current state of a particle processed by Kalman filtering
            @param measurement:   The last measurement used for filter step
            @param state:         The current state containing the uncertainty 

            @return     The updated state
        """
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
        return current_state

class CV_State(object):
    def __init__(self, initial_pos, velocity_guess=15000, pos_var=5, velo_var=10000):
        """
            Initialize a new state of a particle tracked with the constant velocity model

            @param initial_pos:          The initial position of the particle as list [x, y]
            @param velocity_guess=15000: The initial velocity in x direction of a particle  
            @param pos_var=5:            The initial variance of position
            @param velo_var=10000:       The initial variance of velocity   
        """

        # state=[x, v_x, y, v_y]'
        self.state = np.array([[initial_pos[0]], [velocity_guess], [initial_pos[1]], [0]])
        # Prediction covariance matrix
        #scale = 50
        pos_var = 5
        velo_var = 10000
        self.C_p = np.matrix([[pos_var, 0, 0, 0],
                             [0, velo_var, 0, 0],
                             [0, 0, pos_var, 0],
                             [0, 0, 0, velo_var]])
        # Estimate covariance matrix
        self.C_e = np.matrix([[pos_var, 0, 0, 0],
                             [0, velo_var, 0, 0],
                             [0, 0, pos_var, 0],
                             [0, 0, 0, velo_var]])

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