"""Abstract Expert and Expert Type.

Todo:
    * 
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
from enum import Enum, auto

class Expert(ABC):
    """The abstract expert class is a framework for every prediction expert.

    An expert has the basic attributes "type" and "name" to identify it.
    It can be useful to execute different functionality on different types of experts 
    (i.e. RNN training needs different handling than MLP training).

    Every expert should be able to perform basic predictions.

    Attributes:
        type (Expert_Type): The type of the expert
        name (String):      The name of the expert (For evaluation and logging)
        model_path (String): The path to save/load the model
        calibration_separation_regression_var_* (list): 
                            The calibration parameters for the calibration of the uncertainty prediction in separation prediction
    """

    def __init__(self, expert_type, name, model_path = ""):
        """Initialize a new expert.
        
        Args:
            type (Expert_Type): The type of the expert
            name (String):      The name of the expert (For evaluation and logging)
            model_path (String):The path to save or load the model
        """
        self.type = expert_type
        self.name = name
        self.model_path = model_path
        self.calibration_path = os.path.splitext(model_path)[0] + "_calibration.pkl"
        self.calibration_separation_regression_var_spatial = [1, 0]
        self.calibration_separation_regression_var_temporal = [1, 0]

    @abstractmethod
    def train_batch(self, inp, target):
        """Train expert on a batch of training data.
        
        Must return predictions for the input data even if no training is done.

        Args:
            inp (tf.Tensor):    A batch of input tracks
            target (tf.Tensor): A batch of tagets to predict to

        Returns:
            predictions (np.array or tf.Tensor): Predicted positions for training instances
        """
        pass

    @abstractmethod
    def predict_batch(self, inp):
        """Predict a batch of input data for testing.

        Args:
            inp (tf.Tensor): A batch of input tracks

        Returns
            prediction (np.array or tf.Tensor): Predicted positions for training instances
        """
        pass

    @abstractmethod
    def test_batch_separation_prediction(self, **kwargs):
        """Test a batch of input data on separation prediction."""
        pass

    @abstractmethod
    def train_batch_separation_prediction(self, **kwargs):
        """Train a batch of input data on separation prediction."""
        pass

    def ence_calibration_separation(self, predicted_var, target_y, predicted_y, percentage_bin_size = 0.25, domain = "spatial"):
        """Calibrate the uncertainty prediction of the expert in the separation prediction with an ENCE calibration.

        Saves the model after finding new calibration parameters.

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
        """
        # Test plot of calibration. Only for test reasons. This should be deactivated as default.
        RMV_corrected = reg.predict(np.expand_dims(RMV, -1))
        min_RMV = np.min([np.min(RMV),np.min(RMV_corrected)])
        max_RMV = np.max([np.max(RMV),np.max(RMV_corrected)])
        plot_RMV = np.arange(min_RMV, max_RMV, (max_RMV-min_RMV)/RMV.shape[0])
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
        # Print values to csv for thesis.
        plot_output_dict = {}
        plot_output_dict["RMV"] = RMV[::5]
        plot_output_dict["RMSE"] = RMSE[::5]
        plot_output_dict["plot_RMV"] = plot_RMV[::5]
        plot_output_dict["lin_reg"] = reg.predict(np.expand_dims(plot_RMV, -1))[::5]
        plot_output_dict["RMV_corrected"] = RMV_corrected[::5]
        output_df = pd.DataFrame(plot_output_dict)
        output_df.to_csv('{}ENCE_calibration_{}.csv'.format(
            "plot_functions/results/vbc_model/", domain), index=False)
        """
        
    @abstractmethod
    def correct_separation_prediction(self, **kwargs):
        """Correct the uncertainty prediction of the expert with the ENCE calibration."""
        pass

    @abstractmethod
    def save_model(self):
        """Save the model to its model path."""
        pass

    @abstractmethod
    def load_model(self):
        """Load the model and its calibration from its model path."""
        pass

    def save_calibration(self):
        with open(self.calibration_path, 'wb') as f:
            pickle.dump([self.calibration_separation_regression_var_spatial, self.calibration_separation_regression_var_temporal], f)

    def load_calibration(self):
        if os.path.exists(self.calibration_path):
            with open(self.calibration_path, 'rb') as f:
                self.calibration_separation_regression_var_spatial, self.calibration_separation_regression_var_temporal = pickle.load(f)
        else:
            logging.warning("Calibration file for Kalman filter model '{}' does not exist at {}.".format(self.name, self.calibration_path))

    def get_type(self):
        """Return type."""
        return self.type
    
    def change_learning_rate(self, lr_change=1):
        """Change the learning rate of the model optimizer."""
        pass
    
    @abstractmethod
    def get_zero_state(self, batch_size):
        """Return batch of default states."""
        pass

class Expert_Type(Enum):
    """Simple enumeration class for expert types."""
    
    RNN = auto()
    KF = auto()
    MLP = auto()