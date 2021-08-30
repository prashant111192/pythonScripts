# /usr/bin/python3

import os
from typing import TYPE_CHECKING, Tuple
import numpy as np
from glob import glob
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import par
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from scipy.spatial import distance
import time

if TYPE_CHECKING:
    from numpy import ndarray


start_time = time.time()
print(start_time)

pool = mp.Pool(mp.cpu_count()) 




path = os.getcwd()
# set_number = input("Which height is required (Hx)??: ")
set_number = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# 12 fucking arrays...dont have to count all the time
heights_array = [.105, .115, .125, .150, .160, .170, .200, .210, .220, .245, .255, .265]
y_shift_array = np.linspace(1,0.7,100)
average_percent_arr = np.zeros((len(heights_array), len(y_shift_array)))
average_percent_arr = [pool.apply(par.par, args=(path, set_number, heights_array, y_shift_array, idy)) for idy in range(len(y_shift_array))]
pool.close()
average_percent_arr = np.transpose(average_percent_arr)
# average_percent_y = par(path, set_number, heights_array, y_shift_array, average_percent_arr, idy)
# for idy in range(len(y_shift_array)):
    # plot_2_together(shifted_height, sph_arr, None, heights_array[idx], y_shift_array[idy], "position")

np.savetxt(f"./figs2/percent.csv",  average_percent_arr,header = ','.join([str(yy) for yy in y_shift_array]),comments='', delimiter=',')
# np.savetxt(f"./figs2/percent.csv",  average_percent_arr, header = ([y_shift_array]),delimiter=',')
    # plot_graph(data[:,0,0], data[:,1,0], diff, heights_array[i], "difference")
    # print(data.shape)

print("Process finished --- %s seconds ---" % (time.time() - start_time))