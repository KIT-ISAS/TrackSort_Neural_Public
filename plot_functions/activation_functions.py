import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

save_path = dir_path + "/results/"

x = np.arange(-7, 7, 0.1)
y = 1/(1+ np.exp(-x))

#plt.plot(x,y)
#plt.show()

pd_data = pd.DataFrame({'x': x, 'y': y})

pd_data.to_csv(save_path + "sigmoid.csv", index=False)