import numpy as np 
import matplotlib
plt = matplotlib.pyplot

def velocity_plot(aligned_track_data, normalization_constant=1, dt=0.01):
    if normalization_constant != 1:
        aligned_track_data /= normalization_constant
    ma_arr = np.ma.array(aligned_track_data, mask = aligned_track_data==0.0)
    x_velo = (ma_arr[:,1:,0]-ma_arr[:,:-1,0])/dt
    #y_velo = (ma_arr[:,1:,1]-ma_arr[:,:-1,1])/dt
    #x_p = np.arange(0.39,0.78,0.01)
    x_p = np.arange(np.ma.min(ma_arr[:,:,0]),np.ma.max(ma_arr[:,:,0]),0.01)
    x_median_velos = np.zeros([x_p.shape[0]-1,2])
    for i in range(x_p.shape[0]-1):
        x_start = x_p[i]
        x_end = x_p[i+1]
        x_mid = (x_end+x_start)/2
        x_median_velos[i,0] = x_mid
        all_x_velo_in_range = x_velo[np.ma.bitwise_and(ma_arr[:,:-1,0]>=x_start, ma_arr[:,:-1,0]<=x_end)]
        x_median_velos[i,1] = np.ma.median(all_x_velo_in_range)
    
    plt.plot(x_median_velos[:,0], x_median_velos[:,1])
    plt.xlabel('Belt position')
    plt.ylabel('velocity in belt direction [m/s]')
    plt.show()
