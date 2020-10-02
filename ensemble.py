"""The ensemble classes of gating networks for tracking.

An ensemble does not learn the weights based on inputs but assigns the same weights to the experts all the time.

Change log (Please insert your name here if you worked on this file)
    * Created by: Jakob Thumm (jakob.thumm@student.kit.edu)
    * Jakob Thumm 2.10.2020:    Completed documentation.
"""
import numpy as np
import logging
import pickle
import os

from gating_network import GatingNetwork
from expert import Expert_Type

class Simple_Ensemble(GatingNetwork):
    """The simple ensemble gating network weights every expert with the same weight.

    Inherits from GatingNetwork.

    Attributes:
        n_experts (int):    Number of experts
    """

    __metaclass__ = GatingNetwork

    def __init__(self, n_experts, model_path):
        """Initialize a simple ensemble gating network.
        
        Args:
            n_experts (int): Number of experts in expert net
            model_path (String): Path to save or load model
        """
        super().__init__(n_experts, False, "Simple Ensemble", model_path)

    def train_network(self, **kwargs):
        """Simple ensemble needs no training."""
        pass

    def get_weights(self, batch_size):
        """Return an equal weights vector.
        
        The weights sum to 1.
        All Weights are > 0.

        Args:
            batch_size (int):   Number of tracks
        Returns:
            np.array with weights of shape [n_experts, batch_size]
        """
        return (1/self.n_experts) * np.ones([self.n_experts, batch_size])

    def get_masked_weights(self, mask, *args):
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
            mask (np.array):    Mask array with shape [n_experts, n_tracks, track_length]
        Returns:
            np.array with weights of shape mask.shape
        """
        epsilon = 1e-30
        weights = mask / (np.sum(mask, axis=0) + epsilon)
        return weights

class Covariance_Weighting_Ensemble(GatingNetwork):
    """This ensemble gating network weights every expert based on their covariance matrix.

    Inherits from GatingNetwork.

    Attributes:
        n_experts (int):    Number of experts
        weights (np.array): Weights based on covariance
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts, model_path):
        """Initialize a covariance weighting ensemble gating network.
        
        Args:
            n_experts (int): Number of experts in expert net
            model_path (String): Path to save or load model
        """
        self.weights = np.zeros(n_experts)
        super().__init__(n_experts, False, "Covariance Weighting Ensemble", model_path)

    def save_model(self):
        """Save the model weights to self.model_path."""
        folder_path = os.path.dirname(self.model_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filehandler = open(self.model_path, 'wb') 
        pickle.dump(self.weights, filehandler)

    def load_model(self):
        """Load the model weights from self.model_path."""
        try:
            filehandler = open(self.model_path, 'rb') 
            self.weights = pickle.load(filehandler)
        except:
            logging.error("Could not load gating network from path {}".format(self.model_path))

    def train_network(self, target, predictions, masks, **kwargs):
        """Learn the ensemble weights based on the covariance matrix.
        
        Args:
            targets (np.array):     All target values of the given dataset, shape = [n_tracks, track_length, 2]
            predictions (np.array): All predictions for all experts, shape = [n_experts, n_tracks, track_length, 2]
            masks (np.array):       Masks for each expert, shape = [n_experts, n_tracks, track_length]
        Learns:
            self.weights
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

        Args:
            batch_size (int):   Number of tracks
        Returns:
            np.array with weights of shape [n_experts, batch_size]
        """
        weights = np.repeat(np.expand_dims(self.weights, -1), batch_size, axis=-1)
        return weights

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
        batch_weight = self.get_weights(mask.shape[1])
        batch_weight = np.repeat(np.expand_dims(batch_weight, -1), mask.shape[2], axis=-1)
        epsilon = 1e-30
        weights = np.multiply(mask, batch_weight) / (np.sum(np.multiply(mask, batch_weight), axis=0) + epsilon)
        return weights


class SMAPE_Weighting_Ensemble(GatingNetwork):
    """This ensemble gating network weights every expert based on their SMAPE error.

    Inherits from GatingNetwork.

    Attributes:
        n_experts (int):    Number of experts
        weights (np.array): Weights based on covariance
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts, model_path):
        """Initialize a SMAPE weighting ensemble gating network.
        
        Args:
            n_experts (int): Number of experts in expert net
            model_path (String): Path to save or load model
        """
        self.weights = np.zeros(n_experts)
        super().__init__(n_experts, False, "SMAPE Weighting Ensemble", model_path)

    def save_model(self):
        """Save the model weights to self.model_path."""
        folder_path = os.path.dirname(self.model_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filehandler = open(self.model_path, 'wb') 
        pickle.dump(self.weights, filehandler)

    def load_model(self):
        """Load the model weights from self.model_path."""
        try:
            filehandler = open(self.model_path, 'rb') 
            self.weights = pickle.load(filehandler)
        except:
            logging.error("Could not load gating network from path {}".format(self.model_path))

    def train_network(self, target, predictions, masks, expert_types, **kwargs):
        """Learn the ensemble weights based on the SMAPE error.
        
        Calculate the symetric mean percentage error (SMAPE) for every expert.
        Use the softmax function to generate weights from the SMAPE values
        
        Args:
            targets (np.array):     All target values of the given dataset, shape: [n_tracks, track_length, 2]
            predictions (np.array): All predictions for all experts, shape: [n_experts, n_tracks, track_length, 2]
            masks (np.array):       Masks for each expert, shape: [n_experts, n_tracks, track_length]
            expert_types (list):    List of Expert_Types
        Learns:
            self.weights
        """
        n_experts = predictions.shape[0]
        # Duplicate mask to be valid for x_target and y_target and invert mask to fit numpy mask format
        masks = 1 - np.stack([masks, masks], axis=-1)
        # Check if at least one MLP is in experts
        mlp_pos = []
        for i in range(n_experts):
            if expert_types[i] == Expert_Type.MLP:
                mlp_pos.append(i)
        # SMAPE values
        if len(mlp_pos)>0: 
            smape_mlp = np.zeros(n_experts)
            inc_factor = []
        smape = np.zeros(n_experts)
        # Calculate SMAPE of all experts
        for i in range(n_experts):
            # Calculate error of expert i
            masked_prediction = np.ma.array(predictions[i], mask=masks[i])
            masked_target = np.ma.array(target, mask=masks[i])
            sym_err = np.ma.abs(masked_prediction-masked_target)/(np.ma.abs(masked_prediction)+np.ma.abs(masked_target))
            smape[i] = 100 * np.ma.mean(sym_err)
            if len(mlp_pos)>0: 
                masked_prediction = np.ma.array(predictions[i], mask=masks[mlp_pos[0]])
                masked_target = np.ma.array(target, mask=masks[mlp_pos[0]])
                sym_err = np.ma.abs(masked_prediction-masked_target)/(np.ma.abs(masked_prediction)+np.ma.abs(masked_target))
                smape_mlp[i] = 100 * np.ma.mean(sym_err) 
                # Calculate how much the SMAPE increases from MLP_mask to not MLP_mask if this expert is not of type MLP
                if i not in mlp_pos:
                    inc_factor.append(smape[i]/smape_mlp[i])
        # Mean inc_factor
        mean_inc_factor = np.mean(np.array(inc_factor))
        # Increase smape value for all MLPs by the mean_inc_factor
        for idx in mlp_pos:
            smape[idx] *= mean_inc_factor
        # Calculate weights with softmax
        soft_max_weights_1 = np.exp(-smape)/np.sum(np.exp(-smape))
        soft_max_weights_2 = np.exp(-10*smape)/np.sum(np.exp(-10*smape))
        g_inv = 1/smape
        inv_weights = g_inv/np.sum(g_inv)
        g_squared = np.power(g_inv, 2)
        squared_weights = g_squared/np.sum(g_squared)
        g_exp = np.exp(g_inv)
        exp_weights = g_exp/np.sum(g_exp)
        self.weights = squared_weights
        # Logging info
        logging.info("Trained covariance weighting gating network. The resulting weights are: {}".format(self.weights))

    def get_weights(self, batch_size):
        """Return a weight vector.
        
        The weights sum to 1.
        All Weights are > 0.

        Args:
            batch_size (int):   Number of tracks
        Returns:
            np.array with weights of shape [n_experts, batch_size]
        """
        weights = np.repeat(np.expand_dims(self.weights, -1), batch_size, axis=-1)
        return weights

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
        batch_weight = self.get_weights(mask.shape[1])
        batch_weight = np.repeat(np.expand_dims(batch_weight, -1), mask.shape[2], axis=-1)
        epsilon = 1e-30
        weights = np.multiply(mask, batch_weight) / (np.sum(np.multiply(mask, batch_weight), axis=0) + epsilon)
        return weights