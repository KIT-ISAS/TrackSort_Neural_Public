import numpy as np 

def weighting_function(predictions, weights, position_variances = np.array([])):
    """
        @param predictions:         A np array of predictions from multiple experts
        @param weights:             The weights for each expert
        @param position_variances:  Some experts output position variances. 
                                    When the entry of an expert is an empty list, the prediction has no known variance. 
    """
    assert(predictions.shape[0] == weights.shape[0])
    if position_variances.size > 0:
        assert(weights.shape[0] == position_variances.shape[0])

    total_predictions = np.tensordot(predictions, weights, axes=(0,0))
    return total_predictions       