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


pool = mp.Pool(mp.cpu_count()-4) 


path = os.getcwd()
# set_number = input("Which height is required (Hx)??: ")
set_number = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# 12 fucking arrays...dont have to count all the time
heights_array = [.105, .115, .125, .150, .160, .170, .200, .210, .220, .245, .255, .265]
y_shift_array = np.array([-.09833, -.10244, -.10584, -.11152, -.11331, -.11271, -.11165, -.10954, -.10754, -.09875, -.09395, -.08831])
SPH_Name_list = ["CSV0.01.004", "CSV.01.005","CSV.015.004", "CSV.015.005", "CSV.0175.004", "CSV.0175.005", "CSV.02.004", "CSV.02.005"]
h=[0.006928, 0.008660, 0.006928, 0.008660,0.006928 , 0.008660,0.006928, 0.008660]

# for sph_idx in range(len(SPH_Name_list)):

sph_arr0 = par.read_csv_2(f"{path}/data/{SPH_Name_list[0]}")
# _, index0 = par.cKDTree_method(sph_arr0, h[0])
index0 = 0
index1 = 0
index2 = 0
index3 = 0
index4 = 0
index5 = 0
index6 = 0
index7 = 0
sph_arr1 = par.read_csv_2(f"{path}/data/{SPH_Name_list[1]}")
# _, index1 = par.cKDTree_method(sph_arr1, h[0])
sph_arr2 = par.read_csv_2(f"{path}/data/{SPH_Name_list[2]}")
# _, index2 = par.cKDTree_method(sph_arr2, h[2])
sph_arr3 = par.read_csv_2(f"{path}/data/{SPH_Name_list[3]}")
# _, index3 = par.cKDTree_method(sph_arr3, h[3])
sph_arr4 = par.read_csv_2(f"{path}/data/{SPH_Name_list[4]}")
# _, index4 = par.cKDTree_method(sph_arr0, h[4])
sph_arr5 = par.read_csv_2(f"{path}/data/{SPH_Name_list[5]}")
# _, index5 = par.cKDTree_method(sph_arr1, h[5])
sph_arr6 = par.read_csv_2(f"{path}/data/{SPH_Name_list[6]}")
# _, index6 = par.cKDTree_method(sph_arr2, h[6])
sph_arr7 = par.read_csv_2(f"{path}/data/{SPH_Name_list[7]}")
# _, index7 = par.cKDTree_method(sph_arr3, h[7])
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
average_percent_arr= [pool.apply(par.main_2, args=(path, set_number, sph_y0, heights_array, y_shift_array, idx, sph_arr0, sph_arr1, sph_arr2, sph_arr3, sph_arr4, sph_arr5, sph_arr6, sph_arr7, index0, index1, index2, index3, index4, index5, index6, index7, h)) for idx in range(len(y_shift_array))]
# idx = 0
# for i in range(len(y_shift_array)):
#     print(idx)
#     average_percent_arr[idx] = par.main_2(path, set_number, sph_y0, heights_array, y_shift_array, idx, sph_arr0, sph_arr1, sph_arr2, sph_arr3, sph_arr4, sph_arr5, sph_arr6, sph_arr7, index0, index1, index2, index3, index4, index5, index6, index7, h)
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