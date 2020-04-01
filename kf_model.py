"""KF Model and KF State.

Todo:
    * Implement CA Model (and more)
    * (Convert np representation to tensor representation for mixture of experts)
"""

import numpy as np
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

    def __init__(self, name, F, C_w, H, C_v, default_state_options):
        """Initialize a new Kalman filter with given matrices."""
        self.F = F
        self.C_w = C_w
        self.H = H
        self.C_v = C_v
        self.default_state_options = default_state_options
        super().__init__(Expert_Type.KF, name)

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

    def predict_batch(self, inp):
        """Predict a batch of input data for testing.

        Args:
            inp (tf.Tensor): A batch of input tracks

        Returns
            prediction (np.array): Predicted positions for training instances
        """
        pass
    
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