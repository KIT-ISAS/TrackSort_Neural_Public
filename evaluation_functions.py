"""Functions for evaluating models and creating plots.

TODO:
    * Pick n worst predictions instead of dependent on std value.
"""

import logging
import os

from scipy.stats import chi2

import numpy as np
import matplotlib
import pandas as pd
plt = matplotlib.pyplot

def calculate_error_first_and_second_kind(tracks, particle_ids):
    """Calculate the error of first and second kind of the MTT.

    Error of first kind:
        How many tracks have more than one unique particle associated
    Error of second kind:
        How many particles have more than one track associated
    
    Args:
        tracks (list):           List of Track objects
        particle_ids (np.array): Array of particle ids

    Returns:
        Error of first kind:  Value between 1 (100%) and 0
        Error of second kind: Value between 1 (100%) and 0
    """
    sum_first = 0
    sum_second = 0 
    particle_id_dict = dict()
    for _, track in tracks.items():
        unique_particles = track.get_unique_particle_ids()
        if unique_particles.shape[0] > 1:
            sum_first += 1
        for p_id in unique_particles:
            if p_id in particle_id_dict:
                sum_second += 1
            else:
                particle_id_dict[p_id] = True
    # Check if a particle got no track
    for p_id in particle_ids:
        if p_id not in particle_id_dict:
            logging.warning("Particle with id {} was not associated to any track.".format(p_id))
            # Is this correct???
            sum_second += 1
    error_of_first_kind = sum_first/len(tracks)
    logging.info("Error of first kind: {}".format(error_of_first_kind))
    error_of_second_kind = sum_second/len(particle_ids)
    logging.info("Error of second kind: {}".format(error_of_second_kind))
    return error_of_first_kind, error_of_second_kind

def create_boxplot_evaluation(target, predictions, masks, expert_names, result_dir, 
                              normalization_constant = 1, is_mlp_mask=False, no_show = False):
    """Create the data for MSE and MAE boxplots.
    
    Create box plots and data of plots.
    Save plots and data to result directory.
    The normalization_constant changes the plot in y axis.
        If the n_c is >= 100 we assume that you work with pixel data
        If the n_c is < 100 we assume that you work with data in metric units where 1=1m

    Args:
        target (np.array):      Target values
        predictions (np.array): Predicted values
        masks (np.array):       Masks for every expert
        expert_names (list):    Names (String) of each expert
        result_dir (String):    Directory to save the created plot data to
        normalization_constant (double): Value for denormalization
        is_mlp_mask (Boolean):  Is this evaluation with mlp masks or standard?
        no_show (Boolean):      Do not show the figures. The figures will still be saved.
    """
    n_experts = predictions.shape[0]
    assert(len(expert_names) == n_experts)
    # Get mse and mae values
    mse_values, mae_values = calculate_mse_mae(target, predictions, masks)
    # Create box values for every expert
    mse_box_values = {}; mae_box_values = {}
    mse_boxplot_inputs = []
    mae_boxplot_inputs = []
    mse_boxplot_parameters = np.zeros([n_experts, 5]); mae_boxplot_parameters = np.zeros([n_experts, 5])
    for i in range(n_experts):
        compressed_mse_values = normalization_constant**2 * np.ma.reshape(mse_values[i],(mse_values.shape[1]*mse_values.shape[2])).compressed()
        mse_boxplot_inputs.append(compressed_mse_values)
        mse_boxplot_parameters[i] = np.array(get_box_values(compressed_mse_values))
        compressed_mae_values = normalization_constant * np.ma.reshape(mae_values[i],(mae_values.shape[1]*mae_values.shape[2])).compressed()
        mae_boxplot_inputs.append(compressed_mae_values)
        mae_boxplot_parameters[i] = np.array(get_box_values(compressed_mae_values))

    mse_box_values["labels"]=expert_names; mae_box_values["labels"]=expert_names
    mse_box_values["med"]=mse_boxplot_parameters[:,0]; mae_box_values["med"]=mae_boxplot_parameters[:,0]
    mse_box_values["uq"]=mse_boxplot_parameters[:,1]; mae_box_values["uq"]=mae_boxplot_parameters[:,1]
    mse_box_values["lq"]=mse_boxplot_parameters[:,2]; mae_box_values["lq"]=mae_boxplot_parameters[:,2]
    mse_box_values["uw"]=mse_boxplot_parameters[:,3]; mae_box_values["uw"]=mae_boxplot_parameters[:,3]
    mse_box_values["lw"]=mse_boxplot_parameters[:,4]; mae_box_values["lw"]=mae_boxplot_parameters[:,4]
    # Show plot
    plt.figure(figsize=(19.20,10.80), dpi=100)
    plt.boxplot(mse_boxplot_inputs, sym='', labels=expert_names)
    if normalization_constant >= 100:
        plt.ylabel("MSE [px^2]")
    else:
        plt.ylabel("MSE [m^2]")
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.savefig(result_dir + ('mse_box_plot_mlp_maks.pdf' if is_mlp_mask else 'mse_box_plot.pdf')) 
    if not no_show:
        plt.show()
    plt.figure(figsize=(19.20,10.80), dpi=100)
    if normalization_constant >= 100:
        plt.boxplot(mae_boxplot_inputs, sym='', labels=expert_names)
        plt.ylabel("MAE [px]")
        plt.ylim([0, 5])
    else:
        mm_mae_boxplot_inputs = np.array(mae_boxplot_inputs)*1000
        plt.boxplot(mm_mae_boxplot_inputs.T, sym='', labels=expert_names)
        plt.ylabel("MAE [mm]")
        plt.ylim([0, 1])
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.savefig(result_dir + ('mae_box_plot_mlp_maks.pdf' if is_mlp_mask else 'mae_box_plot.pdf'))  
    if not no_show:
        plt.show()

    # Save data to csv via pandas
    mse_df = pd.DataFrame(mse_box_values)
    mae_df = pd.DataFrame(mae_box_values)
    mse_df.to_csv(result_dir + ('mse_box_values_mlp_mask.csv' if is_mlp_mask else 'mse_box_values.csv'), index=False)
    mae_df.to_csv(result_dir + ('mae_box_values_mlp_mask.csv' if is_mlp_mask else 'mae_box_values.csv'), index=False)

def create_boxplot_evaluation_separation_prediction(target, predictions, masks, expert_names, result_dir, 
                              normalization_constant = 1, time_normalization_constant = 22, no_show = False):
    """Create boxplots for the separation prediction.

    Create 2 boxplots:
        spatial error: The first predicted value is the y_nozzle position. 
        temporal error: The second value is the time from the last prediction to the nozzle array.
    
    Save the data to the result dir.

    Args:
        target (np.array):      Target values
        predictions (np.array): Predicted values
        masks (np.array):       Masks for every expert
        expert_names (list):    Names (String) of each expert
        result_dir (String):    Directory to save the created plot data to
        normalization_constant (double): Value for denormalization the y_nozzle value
        time_normalization_constant (double): Value for denormalization the dt_nozzle value
        no_show (Boolean):      Do not show the figures. The figures will still be saved.
    """
    n_experts = predictions.shape[0]
    assert(len(expert_names) == n_experts)
    # Calculate errors
    errors = predictions - np.repeat(target[np.newaxis,:,0:2], n_experts, axis=0)
    # denormalize the errors
    spatial_errors = errors[:,:,0] * normalization_constant
    temporal_errors = errors[:,:,1] * time_normalization_constant
    
    # Create box values for every expert
    spatial_box_values = {}; temporal_box_values = {}
    spatial_boxplot_inputs = []; temporal_boxplot_inputs = []
    spatial_boxplot_parameters = np.zeros([n_experts, 5]); temporal_boxplot_parameters = np.zeros([n_experts, 5])
    for i in range(n_experts):
        # Spatial error
        spatial_error = spatial_errors[i, np.where(masks[i])]
        spatial_boxplot_parameters[i] = np.array(get_box_values(spatial_error))
        spatial_boxplot_inputs.append(spatial_error[0])
        # Temporal error
        temporal_error = temporal_errors[i, np.where(masks[i])]
        temporal_boxplot_parameters[i] = np.array(get_box_values(temporal_error))
        temporal_boxplot_inputs.append(temporal_error[0])
    if normalization_constant < 100:
        spatial_boxplot_parameters *= 1000
    spatial_box_values["labels"]=expert_names; temporal_box_values["labels"]=expert_names
    spatial_box_values["med"]=spatial_boxplot_parameters[:,0]; temporal_box_values["med"]=temporal_boxplot_parameters[:,0]
    spatial_box_values["uq"]=spatial_boxplot_parameters[:,1]; temporal_box_values["uq"]=temporal_boxplot_parameters[:,1]
    spatial_box_values["lq"]=spatial_boxplot_parameters[:,2]; temporal_box_values["lq"]=temporal_boxplot_parameters[:,2]
    spatial_box_values["uw"]=spatial_boxplot_parameters[:,3]; temporal_box_values["uw"]=temporal_boxplot_parameters[:,3]
    spatial_box_values["lw"]=spatial_boxplot_parameters[:,4]; temporal_box_values["lw"]=temporal_boxplot_parameters[:,4]
    # Show temporal plot
    fig = plt.figure(figsize=(19.20,10.80), dpi=100)
    plt.boxplot(temporal_boxplot_inputs, sym='', labels=expert_names)
    plt.ylabel("Temporal deviation [frames]")
    plt.ylim([-1, 1])
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    fig.autofmt_xdate(rotation=60)
    plt.savefig(result_dir + 'temporal_error_box_plot.pdf') 
    if not no_show:
        plt.show()
    # Show spatial plot
    fig = plt.figure(figsize=(19.20,10.80), dpi=100)
    if normalization_constant >= 100:
        plt.boxplot(spatial_boxplot_inputs, sym='', labels=expert_names)
        plt.ylabel("Spatial deviation [px]")
        plt.ylim([-50, 50])
    else:
        for i in range(len(spatial_boxplot_inputs)):
            spatial_boxplot_inputs[i] = spatial_boxplot_inputs[i]*1000
        plt.boxplot(spatial_boxplot_inputs, sym='', labels=expert_names)
        plt.ylabel("Spatial deviation [mm]")
        plt.ylim([-2.8, 2.8])
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    fig.autofmt_xdate(rotation=60)
    plt.savefig(result_dir + 'spatial_error_box_plot.pdf') 
    if not no_show:
        plt.show()

    # Save data to csv via pandas
    mse_df = pd.DataFrame(spatial_box_values)
    mae_df = pd.DataFrame(temporal_box_values)
    mse_df.to_csv(result_dir + 'spatial_error_box_values.csv', index=False)
    mae_df.to_csv(result_dir + 'temporal_error_box_values.csv', index=False)

def create_spatial_outlier_evaluation(seq2seq_inputs, target, predictions, masks, expert_names, result_dir, 
                                      virtual_belt_edge, virtual_nozzle_array, normalization_constant = 1, 
                                      n_errors = 10, no_show = False):
    """Create an evaluation of the biggest outliers in spatial dimension.

    Args:
        seq2seq_inputs (np.array):      Input data in seq2seq format, shape = [n_tracks, track_length, 2] 
        target (np.array):              Target values, shape = [n_tracks]
        predictions (np.array):         Predicted values. The last entry should be the gating network prediction!, shape = [n_experts, n_tracks]
        masks (np.array):               Masks for every expert, shape = [n_experts, n_tracks]
        expert_names (list):            Names (String) of each expert
        result_dir (String):            Directory to save the created plot data to
        virtual_belt_edge (double):     x-Position of virtual belt edge
        virtual_nozzle_array (double):  x-Position of virtual nozzle array
        normalization_constant (double):        Value for denormalization the y_nozzle value
        time_normalization_constant (double):   Value for denormalization the dt_nozzle value
        n_errors (int):                         Number of errors to plot
        no_show (Boolean):                      Do not show the figures. The figures will still be saved.
    """
    n_experts = predictions.shape[0]
    assert(len(expert_names) == n_experts)
    # Calculate errors
    errors = predictions - np.repeat(target[np.newaxis,:], predictions.shape[0], axis=0)
    # denormalize the errors
    spatial_errors = errors[-1] * normalization_constant
    # Find max absolute errors
    abs_errors = np.abs(spatial_errors)
    sorted_idx = np.argsort(abs_errors)
    sorted_idx = sorted_idx[-n_errors:]
    # Denormalize predictions and input points
    seq2seq_inputs = seq2seq_inputs*normalization_constant
    predictions = predictions*normalization_constant
    target = target*normalization_constant
    for idx in sorted_idx:
        input_points = seq2seq_inputs[idx,np.all(seq2seq_inputs[idx]>0, axis=1)]
        plt.figure(figsize=(19.20,10.80), dpi=100) 
        plt.xlim([0, normalization_constant])
        all_y_pos = np.concatenate([input_points[:,1], predictions[:, idx], [target[idx]]])
        plt.ylim([np.min(all_y_pos)-10, np.max(all_y_pos)+10])
        # Draw virtual belt edge and nozzle array
        plt.plot([virtual_belt_edge, virtual_belt_edge],[0, normalization_constant], 'k--')
        plt.plot([virtual_nozzle_array, virtual_nozzle_array],[0, normalization_constant], 'k--')
        # Draw input points 
        plt.scatter(input_points[:,0],input_points[:,1], c='black')
        for i in range(n_experts-1):
            plt.scatter(virtual_nozzle_array, predictions[i, idx], label=expert_names[i])
        plt.scatter(virtual_nozzle_array, predictions[-1, idx], label=expert_names[-1])
        plt.scatter(virtual_nozzle_array, target[idx], label="Target")
        plt.legend()
        if not no_show:
            plt.show()
        stop=0

    stop=0

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
        rmse_value = np.sqrt(np.mean(mse_values[i]))
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

def create_error_region_evaluation(target, predictions, masks, expert_names, result_dir, 
                                   normalization_constant = 1, is_normalized = False, rastering = [10, 10], no_show = False):
    """Create the error region evaluation.
    
    Normalizes the data if it is not normalized.
    Place a raster over the belt region and associate each prediction to a raster field.
    Calculate the median MAE error in each raster field.
    Denormalizes the data.
    Display the median MAE errors on a 2D color plot.
    The normalization_constant changes the plot in y axis.
        If the n_c is >= 100 we assume that you work with pixel data
        If the n_c is < 100 we assume that you work with data in metric units where 1=1m

    Args:
        target (np.array):      Target values
        predictions (np.array): Predicted values
        masks (np.array):       Masks for every expert
        expert_names (list):    Names (String) of each expert
        result_dir (String):    Directory to save the created plot data to
        normalization_constant (double): Value for denormalization
        is_normalized (Boolean): Is the data normalized to [0, 1]?
        rastering (list):       Number of raster fields in x and y dimension
        no_show (Boolean):      Do not show the figures. The figures will still be saved.
    """
    # Calculate MSE and MAE
    _, mae_list = calculate_mse_mae(target, predictions, masks)
    # Create evaluation for each expert individually
    for i in range(predictions.shape[0]):
        # Mask prediction and target
        mask = 1 - np.stack([masks[i], masks[i]], axis=-1)
        masked_prediction = np.ma.array(predictions[i], mask=mask)
        masked_target = np.ma.array(target, mask=mask)
        # The data must be normalized in order to perform the algorithm
        if not is_normalized:
            normalization_constant = np.ma.max(masked_prediction)
            masked_prediction = masked_prediction / normalization_constant
            masked_target = masked_target / normalization_constant

        median_error_map = np.zeros([rastering[1],rastering[0]])
        count_error_map = np.zeros([rastering[1],rastering[0]])
        # Get raster field for every prediction
        idx = np.ma.floor(masked_target[:,:,0] * rastering[0])
        idy = np.ma.floor(masked_target[:,:,1] * rastering[1])
        # Place each error to its raster field
        for x in range(rastering[0]):
            for y in range(rastering[1]):
                ids = (idx==x) & (idy==y)
                errors = mae_list[i, ids]
                median_error_map[y, x] = np.median(errors) * normalization_constant
                count_error_map[y, x] = errors.shape[0]
                # There must be at least 10 predictions in a field to show the error in the plot
                if count_error_map[y, x] < 10:
                    median_error_map[y, x] = np.nan
        # Plot error map
        plt.figure(figsize=(19.20,10.80), dpi=100)
        if normalization_constant >= 100:
            plt.pcolor(median_error_map, cmap="Reds", vmin=0, vmax=3)
        else:
            plt.pcolor(median_error_map*1000, cmap="Reds", vmin=0, vmax=0.5)
        plt.colorbar()
        x_ticks = np.linspace(0, rastering[0], num=5, endpoint=True)
        x_labels = np.round(x_ticks/rastering[0] * normalization_constant, 2)+1
        if normalization_constant >= 100:
            x_labels = np.floor(x_labels)
        plt.xticks(ticks=x_ticks, labels=x_labels)
        plt.xlabel("x")
        y_ticks = np.linspace(0, rastering[1], num=5, endpoint=True)
        y_ticks = y_ticks[::-1]
        y_labels = np.round(y_ticks/rastering[1] * normalization_constant, 2)+1
        if normalization_constant >= 100:
            y_labels = np.floor(y_labels) 
        plt.yticks(ticks=y_ticks, labels=y_labels)
        plt.ylabel("y")
        title = "Median MAE mapped over belt for expert " + expert_names[i]
        plt.title(title)
        plt.savefig(result_dir + "median_mae_map_{}.pdf".format(expert_names[i].replace(" ", "_")))
        if not no_show:
            plt.show()
        np.savetxt(result_dir + "median_mae_map_{}.csv".format(expert_names[i].replace(" ", "_")), median_error_map, delimiter=',')

def create_weight_pos_evaluation(weights, expert_names, result_dir, no_show = False):
    """Plot expert weight over track position.

    Args:
        weights (np.array):     Weights for experts, shape: [n_experts, n_tracks, track_length]
        expert_names (list):    Names (String) of each expert
        result_dir (String):    Directory to save the created plot data to
        no_show (Boolean):      Do not show the figures. The figures will still be saved.
    """
    assert(weights.shape[0]==len(expert_names))
    mask = np.bitwise_or(weights==0, np.isnan(weights))
    weights = np.ma.masked_array(weights, mask)
    mean_weights_per_ts = np.ma.mean(weights, axis=1)
    x = np.arange(0, mean_weights_per_ts.shape[1])
    weights_dict = dict()
    weights_dict['TrackPos'] = x
    # Show plot
    plt.figure(figsize=(19.20,10.80), dpi=100)
    for i in range(mean_weights_per_ts.shape[0]):
        plt.plot(x, mean_weights_per_ts[i], label=expert_names[i])
        weights_dict[expert_names[i]]=mean_weights_per_ts[i]
    plt.ylabel("Weights")
    plt.xlabel("Track index")
    plt.legend()
    plt.savefig(result_dir + 'weight_plot.pdf')
    if not no_show:
        plt.show()

    # Save data to csv via pandas
    weights_df = pd.DataFrame(weights_dict)
    weights_df.to_csv(result_dir + "mean_weights_track_pos.csv", index=False)
    
def create_mean_weight_evaluation(weights, masks, expert_names, result_dir, no_show):
    """Create a plot that shows the mean weight for each expert.

    Args:
        weights (np.array):  Weights for each track for each expert. Shape: [n_experts, n_tracks]
        masks (np.array):    Indicate where the expert was valid. Shape: [n_experts, n_tracks]
        expert_names (list): List of expert names (Strings)
        result_dir (String): Directory to save the results
        no_show (Boolean):   Disable figure pop-up
    """
    weights = np.ma.array(weights, mask=1-masks)
    mean_weights = np.ma.mean(weights, axis=1)
    result_dict = {}
    for i, expert in enumerate(expert_names):
        result_dict[expert] = [mean_weights[i]]
    fig = plt.figure(figsize=(19.20,10.80), dpi=100)
    plt.bar(range(len(expert_names)), mean_weights, tick_label=expert_names)
    fig.autofmt_xdate(rotation=60)
    plt.savefig(result_dir + 'mean_weights_plot.pdf')
    if not no_show:
        plt.show()
    # Save data to csv via pandas
    weights_df = pd.DataFrame(result_dict)
    weights_df.to_csv(result_dir + "mean_weights.csv", index=False)

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
    if len(predictions.shape) >= 3: 
        masks = np.repeat(masks[..., np.newaxis], predictions.shape[-1], axis=-1)
    masks = 1-masks
    # For each expert
    for i in range(predictions.shape[0]):
        # Mask expert prediction
        masked_prediction = np.ma.array(predictions[i], mask=masks[i])
        # Mask target for specific expert
        masked_target = np.ma.array(target, mask=masks[i])
        if len(masked_prediction.shape)>=3:
            masked_mse_pos = ((masked_target - masked_prediction)**2).mean(axis=-1)
            masked_mae_pos = np.ma.abs(masked_target - masked_prediction).mean(axis=-1)
        else:
            masked_mse_pos = (masked_target - masked_prediction)**2
            masked_mae_pos = np.ma.abs(masked_target - masked_prediction)
        # Calculate mean mse error
        #mse_expert = masked_mse_pos.mean(axis=1).mean(axis=0)
        mse_list.append(masked_mse_pos)
        mae_list.append(masked_mae_pos)

    #log_string = 'Mean Squared Error (MSE) for all experts was: \n {}'.format(mse_list)
    #logging.info(log_string)
    return np.ma.array(mse_list), np.ma.array(mae_list)

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
    if len(prediction_1.shape)>1: mask_1 = np.repeat(mask_1[...,np.newaxis], prediction_1.shape[-1], axis=-1)
    mask_1 = 1 - mask_1
    masked_prediction = np.ma.array(prediction_1, mask=mask_1)
    masked_target = np.ma.array(target, mask=mask_1)
    error_1 = masked_target - masked_prediction
    if len(prediction_2.shape)>1: mask_2 = np.repeat(mask_2[...,np.newaxis], prediction_2.shape[-1], axis=-1)
    mask_2 = 1 - mask_2
    masked_prediction = np.ma.array(prediction_2, mask=mask_2)
    masked_target = np.ma.array(target, mask=mask_2)
    error_2 = masked_target - masked_prediction
    if len(prediction_1.shape)>1: 
        # Calculate correlation in x direction
        rho_x = np.ma.sum(np.ma.multiply(error_1[:,:,0], error_2[:,:,0])) / (np.sqrt(np.ma.sum(np.ma.power(error_1[:,:,0],2)))*np.sqrt(np.ma.sum(np.ma.power(error_2[:,:,0],2))))
        # Calculate correlation in y direction
        rho_y = np.ma.sum(np.ma.multiply(error_1[:,:,1], error_2[:,:,1])) / (np.sqrt(np.ma.sum(np.ma.power(error_1[:,:,1],2)))*np.sqrt(np.ma.sum(np.ma.power(error_2[:,:,1],2))))
        # Calculate correlation - normalization to 1
        rho = np.sqrt(1/2 * (rho_x**2 + rho_y**2))
    else:
        # Calculate correlation in single direction
        rho = np.ma.sum(np.ma.multiply(error_1, error_2)) / (np.sqrt(np.ma.sum(np.ma.power(error_1,2)))*np.sqrt(np.ma.sum(np.ma.power(error_2,2))))
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
        plt.figure(figsize=(19.20,10.80), dpi=100)
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

def create_chi_squared_evaluation(target, predictions, masks, expert_names, result_dir, 
                              normalization_constant = 1, time_normalization_constant = 22, no_show = False):
    """Create an evaluation of the predicted error variances.
    Args:
        target (np.array):      Target values
        predictions (np.array): Predicted values [y_nozzle, t_nozzle, s_y, s_t], shape (n_experts=1, n_tracks, 4)
        masks (np.array):       Masks for every expert
        expert_names (list):    Names (String) of each expert
        result_dir (String):    Directory to save the created plot data to
        normalization_constant (double): Value for denormalization the y_nozzle value
        time_normalization_constant (double): Value for denormalization the dt_nozzle value
        no_show (Boolean):      Do not show the figures. The figures will still be saved.
    """
    # Quantile values
    quantiles = np.arange(0,1,0.01)
    # Dictionary for output 
    cdf_results = {}
    cdf_results["quantiles"] = quantiles
    for expert in range(predictions.shape[0]):
        # (y-y_hat)^2
        error = (target[:,0:2]-predictions[expert,:,0:2])**2
        # (y-y_hat)^2/sigma^2
        chi2_sep = error/np.exp(predictions[expert,:,2:4])
        chi2_values = np.sum(chi2_sep,axis=1)
        chi2_values = chi2_values[np.where(masks[expert])]
        # Inverse cdf of chi2 distribution
        chi2_quantiles = chi2.ppf(quantiles, 2)
        # Get the cdf values of chi2_values
        cdf_values = np.zeros(quantiles.shape)
        for i in range(chi2_quantiles.shape[0]):
            cdf_values[i] = np.sum(chi2_values<=chi2_quantiles[i])/chi2_values.shape[0]
        cdf_results[expert_names[expert]] = cdf_values
        # Plot
        plt.figure(figsize=[19.20, 10.80], dpi=100)
        plt.plot(quantiles, cdf_values)
        plt.xlabel("Expected confidence level")
        plt.ylabel("Empirical confidence level")
        plt.savefig(result_dir + 'chi_squared_analysis_{}.pdf'.format(expert_names[expert]))  
        if not no_show:
            plt.show()
        # Chi squared test
        chi2_yt = np.mean(chi2_sep, axis=0)
        total_chi2 = np.sum(chi2_values)
        N = 2*chi2_values.shape[0]
        reduced_chi_squared = total_chi2/N
        logging.info("Reduced chi squared value for expert {} = {}".format(expert_names[expert], reduced_chi_squared)) # Save data to csv via pandas
    
    cdf_results_df = pd.DataFrame(cdf_results)
    cdf_results_df.to_csv(result_dir + 'chi_squared_analysis.csv', index=False)