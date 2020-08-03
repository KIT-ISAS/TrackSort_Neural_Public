"""The weighting function.

Generates a total prediction value from all experts and their weights.
Can incorporate position variances for predictions (e.g. slope values of C_p from Kalman filters).

Todo:
    * Implement variance functionality
    * Output variance --> Incorporate variance in data association with Mahalanoobis distance
"""

import numpy as np 

def weighting_function(predictions, weights, position_variances = np.array([])):
    """Generate single prediction with position and variance from all experts and corresponding weights.

    Args:
        predictions (np.array): A np array of predictions from multiple experts
        weights (np.array):     The weights for each expert and instance
        position_variances (np.array):  Some experts output position variances. 
                                        When the entry of an expert is an empty list, the prediction has no known variance. 

    Returns
        (prediction array, variance array)
    """
    assert(predictions.shape[0] == weights.shape[0])
    if position_variances.size > 0:
        assert(weights.shape[0] == position_variances.shape[0])
        
    # Expand weights in the last dimension by repeating to generate the same weights for x and y.
    # prediction = prediction_1 * weight_1 + prediction_2 * weight_2 + ... + prediction_n * weight_n
    total_predictions = np.sum(predictions * np.repeat(np.expand_dims(weights, -1), 2, axis=-1), axis=0)
    return total_predictions       

def weighting_function_separation(predictions, weights, is_uncertainty_prediction=False):
    """Generate single prediction with position and variance from all experts and corresponding weights.

    For the separation the prediction in temporal and spatial dimension can be weighted differently.

    Args:
        predictions (np.array):         A np array of predictions from multiple experts, 
                                        shape = [n_expert, batch_size, 4] if is_uncertainty_prediction else [n_expert, batch_size, 2]
                                        predictions[:,:,0:2] = mean prediction (first moment)
                                        predictions[:,:,2:4] = log(sigma^2) (log of second central moment)
        weights (np.array):             The weights for each expert and instance, shape = [n_expert, batch_size, 2]
        position_variances (np.array):  Some experts output position variances. 
                                        When the entry of an expert is an empty list, the prediction has no known variance. 

    Returns
        (prediction array, variance array)
    """
    assert(predictions.shape[0] == weights.shape[0])
    # Expand weights in the last dimension by repeating to generate the same weights for x and y.
    # prediction = prediction_1 * weight_1 + prediction_2 * weight_2 + ... + prediction_n * weight_n
    if not is_uncertainty_prediction:
        total_prediction = np.sum(predictions * weights, axis=0)
    else:
        total_prediction = np.zeros(predictions.shape[1:])
        # Weighted mean = First moment of gaussian mixture
        total_prediction[:,:2] = np.sum(predictions[:,:,:2] * weights, axis=0)
        """
        # Calculate the second central moment of the gaussian mixture
        # First calculate the second moment of the gaussian mixture
        e2_GM_0 = np.sum(weights[:,:,0] * (predictions[:,:,0]**2 + np.exp(predictions[:,:,2])), axis=0)
        e2_GM_1 = np.sum(weights[:,:,1] * (predictions[:,:,1]**2 + np.exp(predictions[:,:,3])), axis=0)
        # Then calculate the second central moment of the GM
        c2_GM_0 = e2_GM_0 - total_prediction[:,0]**2
        c2_GM_1 = e2_GM_1 - total_prediction[:,1]**2
        # Lastly, convert the second central moment back to the log format
        total_prediction[:,2] = np.log(c2_GM_0)
        total_prediction[:,3] = np.log(c2_GM_1)
        """
        total_prediction[:,2] = np.log(np.sum(weights[:,:,0] * np.exp(predictions[:,:,2]), axis=0))
        total_prediction[:,3] = np.log(np.sum(weights[:,:,1] * np.exp(predictions[:,:,3]), axis=0))
        # Error handling if every expert has weight = 0
        total_prediction[(np.sum(np.sum(weights, axis=-1),axis=0)==0),0:2]=np.mean(total_prediction[(np.sum(np.sum(weights, axis=-1),axis=0)>0),0:2],axis=0)
        # Very high uncertainty
        total_prediction[(np.sum(np.sum(weights, axis=-1),axis=0)==0),2:]=1e8

    return total_prediction