"""The gating network.

Todo:
    * Implement ME approaches
"""

from abc import ABC, abstractmethod

class GatingNetwork(ABC):
    """The gating network assigns a weight to each expert.

    Implementations can vary from simple ensemble methods to complex Mixture of Experts (ME) methods.

    Attributes:
        n_experts (int):    Number of experts
        name (String):      Name of gating network
    """

    def __init__(self, n_experts, name, model_path):
        """Initialize a gating network.

        Args: 
            n_experts (int): Number of experts in expert net
            name (String):   Name of gating network
            model_path (String): Path to save or load model
        """
        self.n_experts = n_experts
        self.name = name
        self.model_path = model_path

    def save_model(self):
        """Save the model to the model path given in the config file."""
        pass

    def load_model(self):
        """Load the model from the model path given in the config file."""
        pass

    @abstractmethod
    def train_network(self, **kwargs):
        """Train the network."""
        pass

    @abstractmethod
    def get_weights(self, **kwargs):
        """Return the weights for all experts."""
        pass

    @abstractmethod
    def get_masked_weights(self, mask, **kwargs):
        """Return a weight vector for all non masked experts."""
        pass

    def get_name(self):
        """Return name of gating function."""
        return self.name
