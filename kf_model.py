import numpy as np
from abc import ABC, abstractmethod

class KF_Model(ABC):

    def __init__(self, F, C_w, H, C_v, default_state_options):
        """
            Initialize a new Kalman filter with given matrices

            @param F:    The transition matrix
            @param C_w:  The prediction covariance matrix
            @param H:    The measurement matrix
            @param C_v:  The measurement covariance matrix
            @param default_state_options: The options for creating a new state taken from the config file
        """
        self.F = F
        self.C_w = C_w
        self.H = H
        self.C_v = C_v
        self.default_state_options = default_state_options

    def predict(self, current_state):
        """
            Execute prediction step of Kalman filter.
            Changes the input state.

            @param current_state: The current state of a particle processed by Kalman filtering
        """
        ## Check matrix and state dimensions
        assert(self.F.shape[0] == current_state.state.shape[0])

        ## Predict next step
        # Predict state x_p = F * x_e
        current_state.state = self.F.dot(current_state.state)
        # Predict C_p = F * C_e * F' + C_w
        current_state.C_p = np.matmul(np.matmul(self.F, current_state.C_e), self.F.T) + self.C_w

    def update(self, current_state, measurement):
        """
            Execute filtering step of Kalman filter

            @param current_state: The current state of a particle processed by Kalman filtering
            @param measurement:   The last measurement used for filter step
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
    def get_zero_state(self, batch_size):
        """
            Return a list of dummy states of correct size for the corresponding model

            @param batch_size: The size of the dummy list

            @return list of dummy states
        """
        pass


class KF_State(ABC):

    def __init__(self, state, C_p, C_e):
        """
            Initialize a new Kalman filter state (represents one particle)

            @param state: The state of the Kalman filter
            @param C_p:   The state covariance matrix after prediction step
            @param C_e:   The state covariance matrix after update step
        """
        self.state = state
        self.C_p = C_p
        self.C_e = C_e
        # First update step needs to be skiped 
        self.first = True

    @abstractmethod
    def get_pos(self):
        """
            Returns the x,y position of the state

            @return [x; y] as a numpy array
        """
        pass

    @abstractmethod
    def get_v(self):
        """
            Returns the velocity of the state

            @return [vx; vy] as a numpy array
        """
        pass