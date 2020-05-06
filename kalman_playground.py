import numpy as np
from cv_model import *
from ca_model import *

def kalman_playground(aligned_track_data):
    """A little playground for Kalman filter methods."""
    # calc mean vel in first time step

    
    sum_v = 0
    sum_a = 0
    c = 0
    for track in aligned_track_data:
        c = c+1
        sum_v = sum_v + (track[1,0] - track[0,0])
        sum_a += (track[2,0]-track[1,0])-(track[1,0] - track[0,0])
    dt = 0.005
    mean_v = (sum_v/c)/dt
    mean_a = (sum_a/c)/(dt**2)
    var_v_sum = 0
    var_a_sum = 0
    for track in aligned_track_data:
        var_v_sum += ((track[1,0] - track[0,0])/dt - mean_v)**2
        var_a_sum += (((track[2,0]-track[1,0])-(track[1,0] - track[0,0]))/(dt**2) - mean_a)**2
    var_v = var_v_sum/c
    var_a = var_a_sum/c
    """
    cv_model = CV_Model(name = "CV_Model", dt=0.005, s_w=18, s_v=3.6E-7, default_state_options = {})
    # Test CV model
    c = 0
    error = 0
    for track in aligned_track_data:
        cv_state = CV_State(track[0], velocity_guess=1.8, velo_var_x=0.13, velo_var_y = 0.5, pos_var=3.6E-7)
        cv_model.update(cv_state, track[0,:])
        cv_model.predict(cv_state)
        for i in range(1,track.shape[0]):
            pos = track[i,:]
            if np.array_equal(pos,np.array([0, 0])) == False:
                state_pos = [cv_state.get_pos().item(0), cv_state.get_pos().item(1)]
                error = error + np.sqrt((pos[0]-state_pos[0]) ** 2 + (pos[1]-state_pos[1]) ** 2)
                # update
                cv_model.update(cv_state, pos)
                # predict
                cv_model.predict(cv_state)
                c = c+1

    print(error/c)
    """
    ca_model = CA_Model(name="CA_Model", dt=0.005, s_w=1.8E3, s_v=3.6E-7, default_state_options = {})
    # Test CA model
    c = 0
    error = 0
    for track in aligned_track_data:
        ca_state = CA_State(track[0], velocity_guess=1.8, velo_var_x=0.13, velo_var_y = 0.5, pos_var=3.6E-7, accel_guess = 0.0, accel_var_x = 18000, accel_var_y = 180000)
        ca_model.update(ca_state, track[0,:])
        ca_model.predict(ca_state)
        for i in range(1,track.shape[0]):
            pos = track[i,:]
            if np.array_equal(pos,np.array([0, 0])) == False:
                state_pos = [ca_state.get_pos().item(0), ca_state.get_pos().item(1)]
                error = error + np.sqrt((pos[0]-state_pos[0]) ** 2 + (pos[1]-state_pos[1]) ** 2)
                # update
                ca_model.update(ca_state, pos)
                # predict
                ca_model.predict(ca_state)
                c = c+1

    print(error/c)
    stop=0