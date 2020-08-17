"""The gating network.

Todo:
    * Implement ME approaches
"""
import numpy as np
import matplotlib
import logging
import pandas as pd
import os
import pickle
plt = matplotlib.pyplot
from sklearn.linear_model import LinearRegression
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
        self.calibration_path = os.path.splitext(model_path)[0] + "_calibration.pkl"
        # Covariance and correlation matrix of experts
        # Two dimensions: Either (x,y) for tracking or (spatial, temporal) for seperation prediction
        self.C = np.zeros([2, n_experts, n_experts])
        self.corr = np.zeros([2, n_experts, n_experts])
        # ENCE calibration variables
        self.calibration_separation_regression_var_spatial = [1, 0]
        self.calibration_separation_regression_var_temporal = [1, 0]

    def save_model(self):
        """Save the model to the model path given in the config file."""
        pass

    def load_model(self):
        """Load the model from the model path given in the config file."""
        pass

    def train_correlations(self, target, predictions, masks):
        """Train the correlations between the expert prediction errors.

        This is needed to perform the weighting with uncertainty.

        Args:
            target (np.array):      All target values of the given dataset, shape: [n_tracks, 4]
            predictions (np.array): All predictions for all experts, shape: [n_experts, n_tracks, 4]
            masks (np.array):       Masks for each expert, shape: [n_experts, n_tracks]

        Trains:
            self.corr (np.array):  The correlation between each experts prediction error.
                                    Shape: [n_dim (2), n_experts, n_experts]
        """
        # convert masks for numpy
        masks = 1-masks
        for dim in range(2):
            # Calculate covariance between all experts
            # Create covariance matrix
            for i in range(self.n_experts):
                for j in range(self.n_experts):
                    # Calculate error of expert i
                    masked_prediction = np.ma.array(predictions[i,:,dim], mask=masks[i])
                    masked_target = np.ma.array(target[:,dim], mask=masks[i])
                    error_i = masked_target - masked_prediction
                    # Calculate error of expert j
                    masked_prediction = np.ma.array(predictions[j,:,dim], mask=masks[j])
                    masked_target = np.ma.array(target[:,dim], mask=masks[j])
                    error_j = masked_target - masked_prediction
                    # Calculate error covariance in each direction (x and y) seperately
                    # C_ij = mean(error_i * error_j)
                    mult_errors = np.ma.multiply(error_i, error_j)
                    self.C[dim, i, j] = np.ma.mean(mult_errors)
            # Create correlations
            for i in range(self.n_experts):
                for j in range(self.n_experts):
                    self.corr[dim, i, j] = self.C[dim, i, j]/(np.sqrt(self.C[dim, i, i]) * np.sqrt(self.C[dim, j, j]))

    def ence_calibrate(self, predicted_var, target_y, predicted_y, percentage_bin_size = 0.25, domain = "spatial"):
        """Calibrate the uncertainty prediction of the gating network in the separation prediction with an ENCE calibration.

        Args:
            predicted_var (np.array):   The predicted variances of the expert
            target_y (np.array):        The target vector
            predicted_y (np.array):     The prediction vector of the expert
            percentage_bin_size (double): The percentage bin size [0, 1]
            domain (String):            spatial or temporal
        """
        assert(domain == "spatial" or domain == "temporal")
        sorted_indices = np.argsort(predicted_var)
        n_instances = sorted_indices.shape[0]
        bin_size = int(np.floor(n_instances*percentage_bin_size))
        start_ids = np.arange(start=0, stop=n_instances-bin_size, step=1)
        n_bins = start_ids.shape[0]
        RMV = np.zeros(n_bins)
        RMSE = np.zeros(n_bins)
        # Sliding window
        for start_id in start_ids:
            bin_indices = sorted_indices[start_id:start_id+bin_size]
            RMV[start_id] = np.sqrt(np.mean(predicted_var[bin_indices]))
            bin_errors = target_y[bin_indices] - predicted_y[bin_indices]
            RMSE[start_id] = np.sqrt(np.mean(bin_errors**2))
        # Create linear regression for RMV/RMSE plot
        reg = LinearRegression().fit(np.expand_dims(RMV, -1), RMSE)
        if domain == "spatial":
            self.calibration_separation_regression_var_spatial = [reg.coef_[0], reg.intercept_]
        else:
            self.calibration_separation_regression_var_temporal = [reg.coef_[0], reg.intercept_]
        #Test plot of calibration
        """
        RMV_corrected = reg.predict(np.expand_dims(RMV, -1))
        min_RMV = np.min([np.min(RMV),np.min(RMV_corrected)])
        max_RMV = np.max([np.max(RMV),np.max(RMV_corrected)])
        plot_RMV = np.arange(min_RMV, max_RMV, (max_RMV-min_RMV)/1000)
        plt.figure(figsize=[19.20, 10.80], dpi=100)
        plt.plot(RMV, RMSE, '-b', label="ENCE analysis")
        plt.plot(plot_RMV, reg.predict(np.expand_dims(plot_RMV, -1)), '-.b', label="Linear regression of ENCE analysis")
        plt.plot(RMV_corrected, RMSE, '-g', label="Linearly calibrated predictions")
        plt.plot(plot_RMV, plot_RMV, '--k', label="Optimal calibration")
        plt.xlabel("RMV")
        plt.ylabel("RMSE")
        plt.legend()
        plt.title("Calibration analysis for {} prediction of expert {}".format(domain, self.name))
        plt.show()
        """

    def save_calibration(self):
        """Save calibration data to a file.

        This saves the ENCE variables and the covariance / correlation matrices.

        Saves:
            self.calibration_separation_regression_var_spatial
            self.calibration_separation_regression_var_temporal
            self.C
            self.corr
        """
        with open(self.calibration_path, 'wb') as f:
            pickle.dump([self.calibration_separation_regression_var_spatial, 
                        self.calibration_separation_regression_var_temporal,
                        self.C, self.corr], f)

    def load_calibration(self):
        """Load calibration data from file.

        This loads the ENCE variables and the covariance / correlation matrices.

        Loads:
            self.calibration_separation_regression_var_spatial
            self.calibration_separation_regression_var_temporal
            self.C
            self.corr
        """
        if os.path.exists(self.calibration_path):
            with open(self.calibration_path, 'rb') as f:
                self.calibration_separation_regression_var_spatial, \
                    self.calibration_separation_regression_var_temporal, \
                    self.C, self.corr = pickle.load(f)
        else:
            logging.warning("Calibration file for Kalman filter model '{}' does not exist at {}.".format(self.name, self.calibration_path))


    @abstractmethod
    def train_network(self, **kwargs):
        """Train the network."""
        pass

    @abstractmethod
    def get_weights(self, **kwargs):
        """Return the weights for all experts."""
        pass

    @abstractmethod
    def get_masked_weights(self, mask, *args):
        """Return a weight vector for all non masked experts."""
        pass

    def get_masked_weights_and_uncertainty(self, masks, log_variance_predictions, inputs=None):
        """Return an equal weights vector for all non-masked experts and the combined uncertainty.

        The weights sum to 1.
        All Weights are > 0 if the expert is non masked at an instance.
        If the mask value at an instance is 0, the experts weight is 0

        Args:
            masks (np.array):                       Masks of experts, shape: [n_experts, n_tracks]
            log_variance_predictions (np.array):    Log of variances of expert predictions, shape: [n_experts, n_tracks, n_dim]
        Uses:
            self.corr (np.array):                   Correlation between expert predictions, shape: [n_dim, n_experts, n_experts]

        Returns:
            weights: np.array with weights, shape: [n_experts, n_tracks, n_dim]
            uncertainty: np.array containing log(variance), shape: [n_tracks, n_dim]
        """
        weights = self.get_masked_weights(masks)
        variance_predictions = np.exp(log_variance_predictions)
        # var[k,l] = sum_{i} sum_{j} (w[i,k,l]*cov[l,i,j]*w[j,k,l])
        #       - sum_{i} (w[i,k,l]**2 * cov[l,i,i])
        #       + sum_{j} (w[i,k,l]**2 * var_pred[i,k,l])
        combined_var_einsum = np.einsum('ikl,jkl,lij,ikl,jkl->kl', weights, weights, self.corr, np.sqrt(variance_predictions), np.sqrt(variance_predictions))
        combined_log_var = np.log(combined_var_einsum)
        # Error handling if every expert has weight = 0 ==> Very high uncertainty
        combined_log_var[(np.sum(np.sum(weights, axis=-1),axis=0)==0),:]=1e8
        return weights, combined_log_var

    def get_name(self):
        """Return name of gating function."""
        return self.name

    def get_covariance_matrix(self):
        """Return the trained covariance matrix."""
        return self.C

    def get_correlation_matrix(self):
        """Return the trained correlation matrxi."""
        return self.corr
