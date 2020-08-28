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

    def __init__(self, n_experts, is_uncertainty_prediction, model_path):
        """Initialize a simple ensemble gating network."""
        super().__init__(n_experts, is_uncertainty_prediction, "Simple Ensemble", model_path)

    def load_model(self):
        self.load_calibration()

    def train_network(self, **kwargs):
        """Simple ensemble needs no training."""
        pass

    def get_weights(self, batch_size):
        """Not needed for separation prediction."""
        pass

    def get_masked_weights(self, masks, *args):
        """Return an equal weights vector for all non-masked experts.
        
        The weights sum to 1.
        All Weights are > 0 if the expert is non masked at an instance.

        If the mask value at an instance is 0, the experts weight is 0
        
        Returns:
            np.array with weights of shape [masks.shape, 2] -> 2 standing for the two dimensions spatial and temporal
        """
        epsilon = 1e-30
        weights = masks / (np.sum(masks, axis=0) + epsilon)
        weights = np.concatenate((weights[...,np.newaxis], weights[...,np.newaxis]), axis=-1)
        return weights

class Covariance_Weighting_Ensemble_Separation(GatingNetwork):
    """This ensemble gating network weights every expert based on their covariance matrix.

    Attributes:
        n_experts (int):    Number of experts
        weights (np.array): Weights based on covariance
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts, is_uncertainty_prediction, model_path):
        """Initialize a covariance weighting ensemble gating network."""
        self.weights = np.zeros([n_experts, 2])
        super().__init__(n_experts, is_uncertainty_prediction, "Covariance Weighting Ensemble", model_path)

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

    def train_network(self, **kwargs):
        """Train the ensemble.
        
        Args:
            target (np.array):      All target values of the given dataset, shape: [n_tracks, 2]
            predictions (np.array): All predictions for all experts, shape: [n_experts, n_tracks, 2]
            masks (np.array):       Masks for each expert, shape: [n_experts, n_tracks]
        """
        for dim in range(2):
            C = np.matrix(self.C[dim])
            try:
                inv_C = C.I
            except:
                inv_C = np.linalg.pinv(C)
            for i in range(self.n_experts):
                self.weights[i, dim] = np.sum(inv_C[i])/np.sum(inv_C)
            logging.info("Trained covariance weighting gating network for separation. \
                 The resulting weights for dimenstion {} are: {}".format(dim, self.weights[:, dim]))

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

    def get_masked_weights(self, masks, log_variance_predictions=None, *args):
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
            masks (tf.Tensor/np.array):                     Mask array, shape: [n_experts, n_tracks]
            log_variance_predictions (tf.Tensor/np.array):  log of uncertainty prediction of experts, shape: [n_experts, n_tracks, 2]

        Returns:
            np.array with weights, shape: [n_experts, n_tracks, 2]
        """
        assert(masks.shape[0] == self.n_experts)
        if not self.is_uncertainty_prediction:
            batch_weight = np.repeat(np.swapaxes(np.expand_dims(self.weights, -1), 1, 2), masks.shape[1], axis=1)
            #weights = masks / (np.sum(masks, axis=0) + epsilon)
            double_masks = np.concatenate((masks[...,np.newaxis], masks[...,np.newaxis]), axis=-1)
            masked_batch_weight = np.multiply(double_masks, batch_weight)
            epsilon = 1e-30
            weights = masked_batch_weight / (np.sum(masked_batch_weight, axis=0) + epsilon)
        else:
            # We don't use the fix C matrix here.
            # We build our own C matrix with the uncertainty predictions and the precalculated correlations.
            assert(log_variance_predictions is not None)
            weights = np.zeros(log_variance_predictions.shape)
            std_predictions = np.sqrt(np.exp(log_variance_predictions))
            C_dyn = np.einsum('ikl,jkl,lij->ijkl', std_predictions, std_predictions, self.corr)
            for dim in range(2):
                # We have to invert C_dyn for each track. 
                # This could be very time consuming. Took a few seconds for ~5000 tracks.
                for track in range(log_variance_predictions.shape[1]):
                    C_dyn_track = np.matrix(C_dyn[:,:, track, dim])
                    try:
                        inv_C = C_dyn_track.I
                    except:
                        inv_C = np.linalg.pinv(C_dyn_track)
                    for expert in range(self.n_experts):
                        weights[expert, track, dim] = np.sum(inv_C[expert])/np.sum(inv_C)
                    # We need some error handling to prevent ridiculously high weights
                    if np.any(np.abs(weights[:, track, dim]) > 1) or \
                        (np.sum(weights[:,track,dim]<0.95) or np.sum(weights[:,track,dim]>1.05)):
                        weights[:, track, dim] = 1/self.n_experts
        return weights


class SMAPE_Weighting_Ensemble_Separation(GatingNetwork):
    """This ensemble gating network weights every expert based on their SMAPE error.

    Attributes:
        n_experts (int):    Number of experts
        weights (np.array): Weights based on covariance
    """
    
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts, is_uncertainty_prediction, model_path):
        """Initialize a SMAPE weighting ensemble gating network."""
        self.weights = np.zeros([n_experts, 2])
        super().__init__(n_experts, is_uncertainty_prediction, "SMAPE Weighting Ensemble", model_path)

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

        Calculate the symetric mean percentage error (SMAPE) for every expert.
        Use the softmax function to generate weights from the SMAPE values
        
        Args:
            targets (np.array):     All target values of the given dataset, shape: [n_tracks, 2]
            predictions (np.array): All predictions for all experts, shape: [n_experts, n_tracks, 2]
            masks (np.array):       Masks for each expert, shape: [n_experts, n_tracks]
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

    def get_masked_weights(self, masks, *args):
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
            masks (tf.Tensor): Mask array with shape [n_experts, n_tracks, track_length]

        Returns:
            np.array with weights of shape masks.shape
        """
        assert(masks.shape[0] == self.n_experts)
        batch_weight = np.repeat(np.swapaxes(np.expand_dims(self.weights, -1), 1, 2), masks.shape[1], axis=1)
        double_masks = np.concatenate((masks[...,np.newaxis], masks[...,np.newaxis]), axis=-1)
        masked_batch_weight = np.multiply(double_masks, batch_weight)
        epsilon = 1e-30
        weights = masked_batch_weight / (np.sum(masked_batch_weight, axis=0) + epsilon)
        return weights
