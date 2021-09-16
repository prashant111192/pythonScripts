# /usr/bin/python3

from PIVplot import plot_graph
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
y_shift_array = np.array([-.09833, -.10244, -.10584, -.11152, -.11331, -.11271, -.11165, -.10954, -.10754, -.09875, -.09395, -.08831])
SPH_Name_list = ["CSV.01.005", "CSV.02.004", "CSV.0175.004", "CSV0.01.004"]

# for sph_idx in range(len(SPH_Name_list)):

sph_arr0 = par.read_csv_2(f"{path}/{SPH_Name_list[0]}")
sph_arr1 = par.read_csv_2(f"{path}/{SPH_Name_list[1]}")
sph_arr2 = par.read_csv_2(f"{path}/{SPH_Name_list[2]}")
sph_arr3 = par.read_csv_2(f"{path}/{SPH_Name_list[3]}")

sph_y0 = min(sph_arr0[:, 2])
# sph_len = len()
# sph_arr = np.
# print(len(y_shift_array))
# average_percent_arr = np.zeros(len(y_shift_array)*3)
# average_percent_temp = np.zeros(len(heights_array)*3)
# average_percent_no_outliers_temp = np.zeros(len(heights_array))
# average_stdev_no_outliers_temp = np.zeros(len(heights_array))
# average_percent_arr = np.zeros((len(heights_array), len(y_shift_array)))
# average_percent_arr_no_outliers = np.zeros((len(heights_array), len(y_shift_array)))
# average_percent_arr = np.zeros((len(heights_array), len(y_shift_array)))
average_percent_arr= [pool.apply(par.main_2, args=(path, set_number, sph_y0, heights_array, y_shift_array, idx, sph_arr0, sph_arr1, sph_arr2, sph_arr3)) for idx in range(len(y_shift_array))]
# idx = 0
# for i in range(len(y_shift_array)):
#     print(idx)
#     average_percent_arr[idx] = par.main_2(path, set_number, heights_array, y_shift_array, idx, sph_arr)
#     idx = idx+1
# average_percent_arr, average_percent_arr_no_outliers= [pool.apply(par.par, args=(path, set_number, heights_array, y_shift_array, idy)) for idy in range(len(y_shift_array))]
# pool.close()
# average_percent_arr_no_outliers = np.transpose(average_percent_arr_no_outliers)

# average_percent_y = par(path, set_number, heights_array, y_shift_array, average_percent_arr, idy)
# for idy in range(len(y_shift_array)):
    # plot_2_together(shifted_height, sph_arr, None, heights_array[idx], y_shift_array[idy], "position")
label = ([str("{:.2f}".format(yy)) for yy in y_shift_array])
# label = ([str(yy) for yy in y_shift_array])
average_percent_arr = np.array(average_percent_arr)
# average_percent_arr = np.insert(average_percent_arr, 1, heights_array, axis = 1)
print(average_percent_arr)

np.savetxt(f"./figs2/percent.csv",  average_percent_arr[:,[1,2]],header = ','.join([str(yy) for yy in y_shift_array]),comments='', delimiter=',')
np.savetxt(f"./figs2/percent_no_outliers.csv",  average_percent_arr[:,[1,3]],header = ','.join([str(yy) for yy in y_shift_array]),comments='', delimiter=',')
np.savetxt(f"./figs2/stdev_no_outliers.csv",  average_percent_arr[:,[1,4]],header = ','.join([str(yy) for yy in y_shift_array]),comments='', delimiter=',')
# np.savetxt(f"./figs2/percent.csv",  average_percent_arr, header = ([y_shift_array]),delimiter=',')
# plot_graph(data[:,0,0], data[:,1,0], diff, heights_array[i], "difference")
    # print(data.shape)


print("Process finished --- %s seconds ---" % (time.time() - start_time))