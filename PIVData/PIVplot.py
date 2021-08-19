#/usr/bin/python3   

import os
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
# import scipy.interpolate as interp
from scipy.spatial import KDTree
from scipy.spatial import distance

# Get number of csv Files
def find_number_files(path):
    totalNumberFiles = len(glob.glob1(path,"*.txt"))
    return(totalNumberFiles)

# read required CSV(text files)
def read_csv(number_files, set_number, path):
    #preparing the name of the files which is iterated upon (the names of the files start from 1)
    name_file=(path+'H'+str(set_number)+'_'+str(f"{(1):04d}")+'.txt')
    # arr = np.loadtxt(name_file, dtype=float, delimiter=',', skiprows=3, usecols=(0,1,5,10))
    arr = loadAndVel(name_file)
    # arr = velocities(arr)
    # reshape the arr prepared to make it 3d
    arr = arr.reshape(arr.shape[0], arr.shape[1],1)

    # +1 becuase range does not iterate obver the last value
    for i in range(2,number_files+1):
        name_file=(path+'H'+str(set_number)+'_'+str(f"{(i):04d}")+'.txt')
        # arr_temp = np.loadtxt(name_file, dtype=float, delimiter=',', skiprows=3, usecols=(0,1,5,10))
        arr_temp = loadAndVel(name_file)
        arr_temp = arr_temp.reshape(arr_temp.shape[0], arr_temp.shape[1],1)
        # x [m],y [m],u [m/s],v [m/s],vorticity [1/s],magnitude [m/s],divergence [1/s],dcev [1],simple shear [1/s],simple strain [1/s],vector direction [degrees]
        arr = np.append(arr, arr_temp, axis = 2)

    return(arr)

def read_csv() -> Tuple["ndarray", "ndarray"]:
    """Read multiple csv files."""
    # 0_Pos.x[m];1_Pos.y[m];2_Pos.z[m];3_Idp;4_Vel.x[m/s];5_Vel.y[m/s];6_Vel.z[m/s];7_Rhop[kg/m^3];8_Type;
    root_dir = path
    arr = np.loadtxt(root_dir+'csv_0000.csv', dtype=float, delimiter=';', skiprows=5, usecols=(0, 1, 2, 3, 4, 5, 6, 8))
    # arr1 = np.loadtxt(root_dir+'csv_0400.csv', dtype=float, delimiter=';', skiprows=5, usecols=(0, 1, 2, 3, 4, 5, 6, 8))

    # picked 0_x,1_y,2_z, 3_idp 4_vx, 5vy, 6_vz, type
    # extract only the fluids
    low = min(arr[:,0])
    arr = arr[arr[:, 7] == 3]
    arr = np.delete(arr, 7, 1)
    # arr2 = arr2[arr2[:, 7] == 3]
    # arr2 = np.delete(arr2, 7, 1)
    return (arr, low)

        
#reading the csv and computing the directional velocites 
def loadAndVel(name_file):
    arr = np.loadtxt(name_file, dtype=float, delimiter=',', skiprows=3, usecols=(0,1,2,3,5,10))
    # print(arr.shape)
    arr = arr[~np.isnan(arr).any(axis=1)]
    # print(arr.shape)
    # arr = velocities(arr)
    return(arr)

#computing the directional velocites from the angle(given in degrees) and magnitude
def velocities(arr):
    rad_vec = np.vectorize(math.radians)
    cos_vec = np.vectorize(math.cos)
    sin_vec = np.vectorize(math.sin)
    rads = rad_vec(arr[:,3])
    u = arr[:,2]*cos_vec(rads)
    v = arr[:,2]*sin_vec(rads)
    arr = np.insert(arr,2,u,axis=1)
    arr = np.insert(arr,3,v,axis=1)
    return (arr)

def closest_point_distance(piv_arr, sph_arr):
    closest_index = distance.cdist(piv_arr, sph_arr, metric='euclidean').argmin()
    return closest_index

def closest_point_kdtree(piv_arr, sph_arr):
    kdtree = KDTree(sph_arr[:,0:2])
    d, closest_index = kdtree.query(piv_arr[:,0:2])
    return closest_index

# def animated(data):
#     marker_size = 15
#     numframes = data.shape[2]
#     numpoints = data.shape[0]
#     color_data = np.random.random((numframes, numpoints))
#     fig = plt.figure()
#     scat = plt.scatter(data[:,0,0], data[:,1,0], marker_size, c= data[:,4,0])
#     animate = ani.FuncAnimation(fig, update_plot, frames= range(numframes), fargs = (color_data, scat))
#     plt.show()

# def update_plot(i, data, scat):
#     scat.set_array(data[i])
#     return scat, 
    
# def cKDTreeMethod(data, num, timestep):
#     tree = cKDTree(arr[:,])
#     nn_dist, index = tree.query()
#     return (nn_dist, index)

# def interpolate_data(arr):
#     interpolator = interp.CloughTocher2DInterpolator(np.array([x,y]).T, z)

def subtract_plt(sph_arr, piv_arr, i):
    sub_array = piv_arr[:, vel]-sph_arr[i, vel_2]
    percent_arr = (sub_array*100)/piv_arr
    return percent_arr

def plot(x,y,c_):
    marker_size = 15
    plt.scatter(x, y, marker_size, c= c_)
    plt.colorbar()
    # plt.clim(0,0.05)
    plt.gca().set_aspect('equal')
    plt.show()
    return 

def main():
    set_number = input("Which height is required (Hx)??: ")
    path = os.getcwd()
    path = str(path+"/H"+str(set_number)+"/")
    number_files = find_number_files(path)
    #Final array is a 3d array with (points, data, time steps). The data is as follows; x,y,u,v,vel magnitude, vel degree
    data = read_csv(number_files, set_number, path)
    plot(data[:,0,0], data[:,1,0], data[:,4,0])
    # print(data.shape)


if __name__=="__main__":
    main()

