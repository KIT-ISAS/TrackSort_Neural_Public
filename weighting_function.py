import numpy as np 

def weighting_function(predictions, weights, position_variances):
    """
        @param predictions:         A np array of predictions from multiple experts
        @param weights:             The weights for each expert
        @param position_variances:  Some experts output position variances. 
                                    When the entry of an expert is an empty list, the prediction has no known variance. 
    """
    assert(predictions.shape[0] == weights.shape[0])
    assert(weights.shape[0] == position_variances.shape[0])

    total_predictions = np.zeros(predictions.shape[1],predictions.shape[2])
    for j in range(predictions.shape[1]):
        single_prediction = 0
        for i in range(predictions.shape[0]):
            if position_variances[i].isempty() == False:
                