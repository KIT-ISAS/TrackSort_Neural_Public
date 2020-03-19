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
        weights (np.array):     The weights for each expert
        position_variances (np.array):  Some experts output position variances. 
                                        When the entry of an expert is an empty list, the prediction has no known variance. 

    Returns
        (prediction array, variance array)
    """
    assert(predictions.shape[0] == weights.shape[0])
    if position_variances.size > 0:
        assert(weights.shape[0] == position_variances.shape[0])

    total_predictions = np.tensordot(predictions, weights, axes=(0,0))
    return total_predictions       