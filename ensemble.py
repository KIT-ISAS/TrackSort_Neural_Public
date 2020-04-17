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
        super().__init__(n_experts, "Simple Ensemble")

    def get_weights(self, batch_size):
        """Return an equal weights vector.
        
        The weights sum to 1.
        All Weights are > 0.

        Returns:
            np.array with weights of shape [n_experts, batch_size]
        """
        return (1/self.n_experts) * np.ones([self.n_experts, batch_size])

    def get_masked_weights(self, mask):
        """Return an equal weights vector for all non masked experts.
        
        The weights sum to 1.
        All Weights are > 0 if the expert is non masked at an instance.

        If the mask value at an instance is 0, the experts weight is 0.

        example mask arry:
        [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
        --> Expert 6 is only active at position 0.
        
        Returns:
            np.array with weights of shape mask.shape
        """
        weights = mask / np.sum(mask, axis=0)
        # Replace nan values
        weights[np.isnan(weights)] = 0
        return weights