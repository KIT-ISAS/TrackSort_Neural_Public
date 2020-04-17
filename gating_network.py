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

    def __init__(self, n_experts, name):
        """Initialize a gating network.

        Args: 
            n_experts (int): Number of experts in expert net
        """
        self.n_experts = n_experts
        self.name = name

    @abstractmethod
    def get_weights(self, **kwargs):
        """Return the weights for all experts."""
        pass

    def get_name(self):
        """Return name of gating function."""
        return self.name
