"""The ensemble classes of gating networks for separation prediction."""
import numpy as np
import logging
import pickle
import os

from gating_network import GatingNetwork
from expert import Expert_Type

class Simple_Ensemble_Separation(GatingNetwork):
    """The simple ensemble gating network weights every expert with the same weight.

    Attributes:
        n_experts (int):    Number of experts
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts):
        """Initialize a simple ensemble gating network."""
        super().__init__(n_experts, "Simple Ensemble", "")

    def load_model(self):
        self.load_calibration()

    def train_network(self, **kwargs):
        """Simple ensemble needs no training."""
        pass

    def get_weights(self, batch_size):
        """Not needed for separation prediction."""
        pass

    def get_masked_weights(self, mask, *args):
        """Return an equal weights vector for all non masked experts.
        
        The weights sum to 1.
        All Weights are > 0 if the expert is non masked at an instance.

        If the mask value at an instance is 0, the experts weight is 0
        
        Returns:
            np.array with weights of shape [mask.shape, 2] -> 2 standing for the two dimensions spatial and temporal
        """
        epsilon = 1e-30
        weights = mask / (np.sum(mask, axis=0) + epsilon)
        weights = np.concatenate((weights[...,np.newaxis], weights[...,np.newaxis]), axis=-1)
        return weights

class Covariance_Weighting_Ensemble_Separation(GatingNetwork):
    """This ensemble gating network weights every expert based on their covariance matrix.

    Attributes:
        n_experts (int):    Number of experts
        weights (np.array): Weights based on covariance
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts, model_path):
        """Initialize a covariance weighting ensemble gating network."""
        self.weights = np.zeros([n_experts, 2])
        super().__init__(n_experts, "Covariance Weighting Ensemble", model_path)

    def save_model(self):
        folder_path = os.path.dirname(self.model_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filehandler = open(self.model_path, 'wb') 
        pickle.dump(self.weights, filehandler)

    def load_model(self):
        try:
            filehandler = open(self.model_path, 'rb') 
            self.weights = pickle.load(filehandler)
        except:
            logging.error("Could not load gating network from path {}".format(self.model_path))
        self.load_calibration()

    def train_network(self, target, predictions, masks, **kwargs):
        """Train the ensemble.
        
        Args:
            target (np.array):      All target values of the given dataset, shape: [n_tracks, 2]
            predictions (np.array): All predictions for all experts, shape: [n_experts, n_tracks, 2]
            masks (np.array):       Masks for each expert, shape: [n_experts, n_tracks]
        """
        # convert masks for numpy
        masks = 1-masks
        n_experts = predictions.shape[0]
        for dim in range(2):
            C = np.matrix(self.C[dim])
            try:
                inv_C = C.I
            except:
                inv_C = np.linalg.pinv(C)
            for i in range(n_experts):
                self.weights[i, dim] = np.sum(inv_C[i])/np.sum(inv_C)
            logging.info("Trained covariance weighting gating network for separation. \
                 The resulting weights for dimenstion {} are: {}".format(dim, self.weights))

    def get_weights(self, batch_size):
        """Return a weight vector.
        
        The weights sum to 1.
        All Weights are > 0.

        Returns:
            np.array with weights of shape [n_experts, batch_size]
        """
        """weights = np.repeat(np.expand_dims(self.weights, -1), batch_size, axis=-1)
        return weights"""
        pass

    def get_masked_weights(self, mask, *args):
        """Return a weights vector for all non masked experts.
        
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
            mask (tf.Tensor): Mask array with shape [n_experts, n_tracks]

        Returns:
            np.array with weights of shape mask.shape
        """
        assert(mask.shape[0] == self.n_experts)
        batch_weight = np.repeat(np.swapaxes(np.expand_dims(self.weights, -1), 1, 2), mask.shape[1], axis=1)
        #weights = mask / (np.sum(mask, axis=0) + epsilon)
        double_mask = np.concatenate((mask[...,np.newaxis], mask[...,np.newaxis]), axis=-1)
        masked_batch_weight = np.multiply(double_mask, batch_weight)
        epsilon = 1e-30
        weights = masked_batch_weight / (np.sum(masked_batch_weight, axis=0) + epsilon)
        return weights


class SMAPE_Weighting_Ensemble_Separation(GatingNetwork):
    """This ensemble gating network weights every expert based on their SMAPE error.

    Attributes:
        n_experts (int):    Number of experts
        weights (np.array): Weights based on covariance
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts, model_path):
        """Initialize a SMAPE weighting ensemble gating network."""
        self.weights = np.zeros([n_experts, 2])
        super().__init__(n_experts, "SMAPE Weighting Ensemble", model_path)

    def save_model(self):
        folder_path = os.path.dirname(self.model_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filehandler = open(self.model_path, 'wb') 
        pickle.dump(self.weights, filehandler)

    def load_model(self):
        try:
            filehandler = open(self.model_path, 'rb') 
            self.weights = pickle.load(filehandler)
        except:
            logging.error("Could not load gating network from path {}".format(self.model_path))
        self.load_calibration()

    def train_network(self, target, predictions, masks, expert_types, **kwargs):
        """Train the ensemble.

        Calculate the symetric mean percentage error (SMAPE) for every expert.
        Use the softmax function to generate weights from the SMAPE values
        
        Args:
            targets (np.array):     All target values of the given dataset, shape: [n_tracks, 2]
            predictions (np.array): All predictions for all experts, shape: [n_experts, n_tracks, 2]
            masks (np.array):       Masks for each expert, shape: [n_experts, n_tracks]
            expert_types (list):    List of Expert_Types
        """
        n_experts = predictions.shape[0]
        # convert masks for numpy
        masks = 1-masks
        
        for dim in range(2):
            smape = np.zeros(n_experts)
            # Calculate SMAPE of all experts
            for i in range(n_experts):
                # Calculate error of expert i
                masked_prediction = np.ma.array(predictions[i, :, dim], mask=masks[i])
                masked_target = np.ma.array(target[:, dim], mask=masks[i])
                sym_err = np.ma.abs(masked_prediction-masked_target)/(np.ma.abs(masked_prediction)+np.ma.abs(masked_target))
                smape[i] = 100 * np.ma.mean(sym_err)

            # Calculate weights with softmax
            soft_max_weights_1 = np.exp(-smape)/np.sum(np.exp(-smape))
            soft_max_weights_2 = np.exp(-10*smape)/np.sum(np.exp(-10*smape))
            g_inv = 1/smape
            inv_weights = g_inv/np.sum(g_inv)
            g_squared = np.power(g_inv, 2)
            squared_weights = g_squared/np.sum(g_squared)
            g_exp = np.exp(g_inv)
            exp_weights = g_exp/np.sum(g_exp)
            self.weights[:, dim] = squared_weights
            # Logging info
            logging.info("Trained SMAPE weighting gating network for separation prediction on dimension {}. The resulting weights are: {}".format(dim, self.weights))

    def get_weights(self, batch_size):
        """Return a weight vector.
        
        The weights sum to 1.
        All Weights are > 0.

        Returns:
            np.array with weights of shape [n_experts, batch_size]
        """
        pass

    def get_masked_weights(self, mask, *args):
        """Return a weights vector for all non masked experts.
        
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
            mask (tf.Tensor): Mask array with shape [n_experts, n_tracks, track_length]

        Returns:
            np.array with weights of shape mask.shape
        """
        assert(mask.shape[0] == self.n_experts)
        batch_weight = np.repeat(np.swapaxes(np.expand_dims(self.weights, -1), 1, 2), mask.shape[1], axis=1)
        double_mask = np.concatenate((mask[...,np.newaxis], mask[...,np.newaxis]), axis=-1)
        masked_batch_weight = np.multiply(double_mask, batch_weight)
        epsilon = 1e-30
        weights = masked_batch_weight / (np.sum(masked_batch_weight, axis=0) + epsilon)
        return weights
