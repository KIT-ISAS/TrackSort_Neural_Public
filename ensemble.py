"""The ensemble classes of gating networks.

Todo:
    * Implement weighted ensembles
"""
import numpy as np
import logging

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

    def train_network(self, **kwargs):
        """Simple ensemble needs no training."""
        pass

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
        epsilon = 1e-30
        weights = mask / (np.sum(mask, axis=0) + epsilon)
        return weights

class Covariance_Weighting_Ensemble(GatingNetwork):
    """This ensemble gating network weights every expert based on their error on a training dataset.

    Attributes:
        n_experts (int):    Number of experts
        weights (np.array): Weights based on covariance
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts):
        """Initialize a covariance weighting ensemble gating network."""
        self.weights = np.zeros(n_experts)
        super().__init__(n_experts, "Covariance Weighting Ensemble")

    def train_network(self, target, predictions, masks, **kwargs):
        """Train the ensemble.
        
        Args:
            targets (np.array):     All target values of the given dataset, shape: [n_tracks, track_length, 2]
            predictions (np.array): All predictions for all experts, shape: [n_experts, n_tracks, track_length, 2]
            masks (np.array):       Masks for each expert, shape: [n_experts, n_tracks, track_length]
        """
        n_experts = predictions.shape[0]
        # Duplicate mask to be valid for x_target and y_target and invert mask to fit numpy mask format
        masks = 1 - np.stack([masks, masks], axis=-1)
        # Covariance matrix C
        C = np.matrix(np.zeros([n_experts, n_experts]))
        # Calculate covariance between all experts
        for i in range(n_experts):
            for j in range(n_experts):
                # Calculate error of expert i
                masked_prediction = np.ma.array(predictions[i], mask=masks[i])
                masked_target = np.ma.array(target, mask=masks[i])
                error_i = masked_target - masked_prediction
                # Calculate error of expert j
                masked_prediction = np.ma.array(predictions[j], mask=masks[j])
                masked_target = np.ma.array(target, mask=masks[j])
                error_j = masked_target - masked_prediction
                # Calculate error covariance in each direction (x and y) seperately
                # C_ij = mean(error_i * error_j)
                mult_errors = np.ma.multiply(error_i, error_j)
                C[i,j] = np.ma.mean(mult_errors)

        inv_C = C.I
        for i in range(n_experts):
            self.weights[i] = np.sum(inv_C[i])/np.sum(inv_C)
        logging.info("Trained covariance weighting gating network. The resulting weights are: {}".format(self.weights))

    def get_weights(self, batch_size):
        """Return a weight vector.
        
        The weights sum to 1.
        All Weights are > 0.

        Returns:
            np.array with weights of shape [n_experts, batch_size]
        """
        weights = np.repeat(np.expand_dims(self.weights, -1), batch_size, axis=-1)
        return weights

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
        
        Args:
            mask (np.array): Mask array with shape [n_experts, batch_size]

        Returns:
            np.array with weights of shape mask.shape
        """
        assert(mask.shape[0] == self.n_experts)
        batch_weight = self.get_weights(mask.shape[1])
        batch_weight = np.repeat(np.expand_dims(batch_weight, -1), mask.shape[2], axis=-1)
        epsilon = 1e-30
        weights = np.multiply(mask, batch_weight) / (np.sum(np.multiply(mask, batch_weight), axis=0) + epsilon)
        return weights