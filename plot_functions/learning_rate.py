import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 


def error_function(x):
    #y = x**2
    y = 3/2 * np.power(x,4) + 7 * np.power(x,3) + 9 * np.power(x,2)
    return y

personal_save_path = "/home/jakob/Documents/Masterarbeit/Ausarbeitung/Masterthesis/tikz/data/learning_rate/"
use_personal_path = False

start_point = -2.7
eta = 0.5
x = [start_point]
y = [error_function(x[0])]
d = 1
for i in range(10):
    x.append(x[i]+d*eta)
    y.append(error_function(x[i]+d*eta))
    if y[i+1] > y[i]:
        d *= -1

start_point = -2.7
eta = 0.1
x_l = [start_point]
y_l = [error_function(x[0])]
d = 1
for i in range(10):
    x_l.append(x_l[i]+d*eta)
    y_l.append(error_function(x_l[i]+d*eta))
    if y_l[i+1] > y_l[i]:
        d *= -1 
x1 = np.arange(-3,3,0.01)
plt.plot(x1, error_function(x1))
plt.plot(x,y)
plt.plot(x_l, y_l)
plt.xlim([-3,1])
plt.ylim([0,10])
plt.show()

dir_path = os.path.dirname(os.path.realpath(__file__))

if use_personal_path:
    save_path=personal_save_path
else:
    save_path = dir_path + "/results/learning_rate/"

folder_path = os.path.dirname(save_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

pd_data = pd.DataFrame({'x': x1, 'y': error_function(x1)})
pd_data.to_csv(save_path + "error_curve.csv", index=False)
pd_data = pd.DataFrame({'x': x, 'y': y})
pd_data.to_csv(save_path + "high_learning_rate.csv", index=False)
pd_data = pd.DataFrame({'x': x_l, 'y': y_l})
pd_data.to_csv(save_path + "low_learning_rate.csv", index=False)


