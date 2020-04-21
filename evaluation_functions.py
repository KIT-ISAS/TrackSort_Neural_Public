"""Functions for evaluating models and creating plots.

TODO:
    * Pick n worst predictions instead of dependent on std value.
"""

import logging
import os

import numpy as np
import matplotlib
import pandas as pd
plt = matplotlib.pyplot

def create_boxplot_evaluation(target, predictions, masks, expert_names, result_dir, normalization_constant = 1, is_mlp_mask=False):
    """Create the data for MSE and MAE boxplots.
    
    Create box plots and data of plots.
    Save plots and data to result directory.

    Args:
        target (np.array):      Target values
        predictions (np.array): Predicted values
        masks (np.array):       Masks for every expert
        expert_names (list):    Names (String) of each expert
        result_dir (String):    Directory to save the created plot data to
        normalization_constant (double): Value for denormalization
        is_mlp_mask (Boolean):  Is this evaluation with mlp masks or standard?
    """
    assert(len(expert_names) == predictions.shape[0])
    # Get mse and mae values
    mse_values, mae_values = calculate_mse_mae(target, predictions, masks)
    # Create box values for every expert
    mse_box_values = {}; mae_box_values = {}
    mse_boxplot_inputs = []
    mae_boxplot_inputs = []
    for i in range(mse_values.shape[0]):
        compressed_mse_values = normalization_constant**2 * np.ma.reshape(mse_values[i],(mse_values.shape[1]*mse_values.shape[2])).compressed()
        mse_box_values[expert_names[i]] = get_box_values(compressed_mse_values)
        compressed_mae_values = normalization_constant * np.ma.reshape(mae_values[i],(mae_values.shape[1]*mae_values.shape[2])).compressed()
        mae_box_values[expert_names[i]] = get_box_values(compressed_mae_values)
        mse_boxplot_inputs.append(compressed_mse_values)
        mae_boxplot_inputs.append(compressed_mae_values)

    # Check if result folder exists and create it if not.
    save_path = os.path.dirname(result_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Show plot
    plt.figure()
    plt.boxplot(mse_boxplot_inputs, sym='', labels=expert_names)
    plt.ylabel("MSE")
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.savefig(result_dir + ('mse_box_plot_mlp_maks.pdf' if is_mlp_mask else 'mse_box_plot.pdf')) 
    plt.show()
    plt.boxplot(mae_boxplot_inputs, sym='', labels=expert_names)
    plt.ylabel("MAE")
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.savefig(result_dir + ('mae_box_plot_mlp_maks.pdf' if is_mlp_mask else 'mae_box_plot.pdf'))  
    plt.show()

    # Save data to csv via pandas
    mse_df = pd.DataFrame(mse_box_values)
    mae_df = pd.DataFrame(mae_box_values)
    mse_df.to_csv(result_dir + ('mse_box_values_mlp_mask.csv' if is_mlp_mask else 'mse_box_values.csv'), index=False)
    mae_df.to_csv(result_dir + ('mae_box_values_mlp_mask.csv' if is_mlp_mask else 'mae_box_values.csv'), index=False)

def get_box_values(data):
    """Obtain all box plot values from a set of numpy data.
    
    Args:
        data: numpy array

    Returns
        [median, upper_quartile, lower_quartile, upper_whisker, lower_whisker]
    """
    median = np.median(data)
    lower_quartile, upper_quartile = np.percentile(data, (25, 75))
    iqr = upper_quartile - lower_quartile
    upper_whisker = np.max(data[data<=upper_quartile+1.5*iqr])
    lower_whisker = np.min(data[data>=lower_quartile-1.5*iqr])
    return [median, upper_quartile, lower_quartile, upper_whisker, lower_whisker]

def calculate_mse_mae(target, predictions, masks):
    """Calculate the Mean Squared Error and Mean Absolut Error for each expert.

    Args:
        target (np.array):      Target values
        predictions (np.array): Predicted values
        masks (np.array):       Masks for every expert

    Returns:
        MSE (np.array): One mse value per expert prediction (Same size as masks)
        MAE (np.array): One mae value per expert prediction (Same size as masks)
    """
    mse_list = []
    mae_list = []
    # Duplicate mask to be valid for x_target and y_target and invert mask to fit numpy mask format
    masks = 1 - np.stack([masks, masks], axis=-1)
    # For each expert
    for i in range(predictions.shape[0]):
        # Mask expert prediction
        masked_prediction = np.ma.array(predictions[i], mask=masks[i])
        # Mask target for specific expert
        masked_target = np.ma.array(target, mask=masks[i])
        masked_mse_pos = ((masked_target - masked_prediction)**2).mean(axis=2)
        masked_mae_pos = np.ma.abs(masked_target - masked_prediction).mean(axis=2)
        # Calculate mean mse error
        #mse_expert = masked_mse_pos.mean(axis=1).mean(axis=0)
        mse_list.append(masked_mse_pos)
        mae_list.append(masked_mae_pos)

    #log_string = 'Mean Squared Error (MSE) for all experts was: \n {}'.format(mse_list)
    #logging.info(log_string)
    return np.ma.array(mse_list), np.ma.array(mae_list)

def create_diversity_evaluation(target, predictions, masks, expert_names, result_dir, is_mlp_mask=False):
    """Create the data for diversity measurement comparison.

    Save data to result directory.

    Args:
        target (np.array):      Target values
        predictions (np.array): Predicted values
        masks (np.array):       Masks for every expert
        expert_names (list):    Names (String) of each expert
        result_dir (String):    Directory to save the created plot data to
        is_mlp_mask (Boolean):  Is this evaluation with mlp masks or standard?
    """
    n_experts = len(expert_names)
    assert(predictions.shape[0] == n_experts)
    ## Disagreement Measure (D)
    # Calculate difference between forecaster and target -> calculate Root Mean Squared Error for each forecaster
    mse_values, mae_values = calculate_mse_mae(target, predictions, masks)
    rmse_values = []
    error_values = np.ma.zeros(mae_values.shape)
    for i in range(n_experts):
        rmse_value = np.sqrt(mse_values[i].mean(axis=1).mean(axis=0))
        # Determine when the forecaster made an error.
        # Error := abs(y_prediction - y_target) > rmse
        error_values[i] = mae_values[i]>rmse_value
        rmse_values.append(rmse_value)
    
    # Calculate error table for every expert combination
    disagreement_measures = dict()
    disagreement_measures['Expert names'] = expert_names
    double_fault = dict()
    double_fault['Expert names'] = expert_names
    correlation_coefficients = dict()
    correlation_coefficients['Expert names'] = expert_names
    #              M_i correct | M_i error
    # M_j correct       N11         N01
    # M_j error         N10         N00
    for i in range(n_experts):
        disagreement_measure_vec = np.zeros(n_experts)
        double_fault_vec = np.zeros(n_experts)
        correlation_coefficient_vec = np.zeros(n_experts)
        for j in range(n_experts):
            correlation_coefficient_vec[j] = calculate_correlation_coefficient(target, predictions[i], predictions[j], masks[i], masks[j])
            N11 = np.ma.sum((error_values[i] == 0) & (error_values[j] == 0))
            N01 = np.ma.sum((error_values[i] == 1) & (error_values[j] == 0))
            N10 = np.ma.sum((error_values[i] == 0) & (error_values[j] == 1))
            N00 = np.ma.sum((error_values[i] == 1) & (error_values[j] == 1))
            disagreement_measure_vec[j] = (N01 + N10) / (N00 + N10 + N01 + N11)
            double_fault_vec[j] = N00 / (N00 + N10 + N01 + N11)
        disagreement_measures[expert_names[i]] = disagreement_measure_vec
        double_fault[expert_names[i]] = double_fault_vec
        correlation_coefficients[expert_names[i]] = correlation_coefficient_vec

    disagreement_measures_df = pd.DataFrame(disagreement_measures)
    double_fault_df = pd.DataFrame(double_fault)
    correlation_coefficient_df = pd.DataFrame(correlation_coefficients)
    disagreement_measures_df.to_csv(result_dir + ('disagreement_measures_mlp_mask.csv' if is_mlp_mask else 'disagreement_measures.csv'), index=False)
    double_fault_df.to_csv(result_dir + ('double_fault_mlp_mask.csv' if is_mlp_mask else 'double_fault.csv'), index=False)
    correlation_coefficient_df.to_csv(result_dir + ('correlation_coefficients_mlp_mask.csv' if is_mlp_mask else 'correlation_coefficients.csv'), index=False)

def calculate_correlation_coefficient(target, prediction_1, prediction_2, mask_1, mask_2):
    """Calculate the correlation coefficient between two predictions.

    Calculates pearsons correlation coefficient in x and y direction (rho_x, rho_y).
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    The total correlation is rho = sqrt(1/2 * (rho_x^2 + rho_y^2))

    Args: 
        target (np.array):      Target values
        prediction_1, prediction_2 (np.array): Predicted values
        mask_1, mask_2 (np.array): Masks for predictions

    Returns:
        Correlation coefficient (double)
    """
    # Calculate errors
    mask_1 = 1 - np.stack([mask_1, mask_1], axis=-1)
    masked_prediction = np.ma.array(prediction_1, mask=mask_1)
    masked_target = np.ma.array(target, mask=mask_1)
    error_1 = masked_target - masked_prediction
    mask_2 = 1 - np.stack([mask_2, mask_2], axis=-1)
    masked_prediction = np.ma.array(prediction_2, mask=mask_2)
    masked_target = np.ma.array(target, mask=mask_2)
    error_2 = masked_target - masked_prediction
    # Calculate correlation in x direction
    rho_x = np.ma.sum(np.ma.multiply(error_1[:,:,0], error_2[:,:,0])) / (np.sqrt(np.ma.sum(np.ma.power(error_1[:,:,0],2)))*np.sqrt(np.ma.sum(np.ma.power(error_2[:,:,0],2))))
    # Calculate correlation in y direction
    rho_y = np.ma.sum(np.ma.multiply(error_1[:,:,1], error_2[:,:,1])) / (np.sqrt(np.ma.sum(np.ma.power(error_1[:,:,1],2)))*np.sqrt(np.ma.sum(np.ma.power(error_2[:,:,1],2))))
    # Calculate correlation - normalization to 1
    rho = np.sqrt(1/2 * (rho_x**2 + rho_y**2))
    return rho    

def find_worst_predictions(target, predictions, mask_value):
    """Find the worst predictions for each expert.
    
    Plot the predictions with the highest squared error.

    Args:
        target (np.array):      Target values
        predictions (np.array): Predicted values
        mask_value (list):      Mask values for targets
    """
    # Find end of tracks and save them to a mask
    mask = np.equal(target, mask_value)
    masked_target = np.ma.array(target, mask=mask)
    
    # For each expert
    for i in range(len(predictions)):
        masked_prediction = np.ma.array(predictions[i], mask=mask)
        masked_mse_pos = ((masked_target - masked_prediction)**2).mean(axis=2)
        #masked_mse_pos = np.ma.array(mse_pos, mask=mask)
        # Calculate mean mse error
        mse_expert = masked_mse_pos.mean(axis=1).mean(axis=0)
        # Calculate standard deviation
        std_expert = masked_mse_pos.std(axis=1).mean(axis=0)
        # Find errors > mean + x * std
        threshold = mse_expert + 1 * std_expert
        greater_array = np.ma.greater(masked_mse_pos,threshold)
        track, pos = np.ma.where(greater_array)
        # Most errors occur in the first time step due to high uncertainty in the state. We can filter those out.
        greater_zero_pos_pos = np.where(np.greater(pos, 0))
        greater_zero_pos = pos[greater_zero_pos_pos]
        greater_zero_track = track[greater_zero_pos_pos]
        stop=0
        
        # Plot all non zero errors with whole track
        track_id = -1
        for j in range(greater_zero_pos.shape[0]):
            # If new track, plot track
            if greater_zero_track[j].item() != track_id:
                track_id = greater_zero_track[j].item()
                plt.plot(masked_target[track_id, :, 0], masked_target[track_id, :, 1], marker = 'o', color = [.6, .6, .6], zorder=1)
                plt.scatter(masked_prediction[track_id, :, 0], masked_prediction[track_id, :, 1], marker = 'x', color = 'green', zorder=5)
            # Draw missed prediction
            plt.scatter(masked_prediction[track_id, greater_zero_pos[j], 0], masked_prediction[track_id, greater_zero_pos[j], 1], 
                    marker = 'x', color = 'red', zorder=10)
        plt.show()
        stop=0
        #mse.append(mse_expert)
    