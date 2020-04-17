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