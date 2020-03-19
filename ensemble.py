"""The ensemble classes of gating networks.

Todo:
    * Implement weighted ensembles
"""
import numpy as np

from gating_network import GatingNetwork

class Simple_Ensemble(GatingNetwork):
    """The simple ensemble gating network weights every expert with the same weight.

    Attributes:
        n_experts (int):    Number of experts
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts):
        """Initialize a simple ensemble gating network."""
        super().__init__(n_experts)

    def get_weights(self):
        """Return a equal weights.
        
        The weights sum to 1.
        All Weights are > 0.

        Returns:
            np.array with weights of shape [n_experts, 1]
        """
        return (1/self.n_experts) * np.ones([self.n_experts,1])