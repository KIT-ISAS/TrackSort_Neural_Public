import numpy as np

class CV_Model(object):
    

    def __init__(self, dt=0.005, s_w=1):
        self.F = np.array([[1, dt, 0, 0],
                           [0,  1, 0, 0],
                           [0,  0, 1, dt],
                           [0,  0, 0, 1]])
        self.C_w = np.array(s_w*[[pow(dt,3)/3, pow(dt,2)/2,           0,           0],
                                [pow(dt,2)/2,          dt,           0,           0],
                                [          0,           0, pow(dt,3)/3, pow(dt,2)/2],
                                [          0,           0, pow(dt,2)/2,          dt]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

    def predict(self, measurement, state):
        """
            Execute prediction and filtering step of Kalman filter

            @param measurement: The current measurement used for filter step
            @param state:       The current state containing the uncertainty 

            @return     The updated state
        """
# predict
"""
allTracks(trackNo).FullState=predictionParam.F*allTracks(trackNo).FullState;
allTracks(trackNo).FullStateCov=predictionParam.F*allTracks(trackNo).FullStateCov*predictionParam.F'+predictionParam.Q;
allTracks(trackNo).Position=allTracks(trackNo).FullState([1,dimState/2+1]);
"""