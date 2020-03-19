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
    """

    def __init__(self, n_experts):
        """Initialize a gating network.

        Args: 
            n_experts (int): Number of experts in expert net
        """
        self.n_experts = n_experts

    @abstractmethod
    def get_weights(self, **kwargs):
        """Return the weights for all experts."""
        pass

