#/usr/bin/python3   

import os
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import time

# Get number of csv Files
def find_number_files(path):
    totalNumberFiles = len(glob.glob1(path,"*.txt"))
    # print(path)
    # print ((totalNumberFiles))
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

def plot(data):
    marker_size = 15
    plt.scatter(data[:,0,0], data[:,1,0], marker_size, c= data[:,4,0])
    plt.colorbar()
    plt.clim(0,0.05)
    plt.gca().set_aspect('equal')
    plt.show()

    plt.close()
    time.sleep(0.5)
    plt.scatter(data[:,0,60], data[:,1,60], marker_size, c= data[:,4,60])
    plt.colorbar()
    plt.clim(0,0.05)
    plt.gca().set_aspect('equal')
    plt.show()
    plt.close()
def main():
    set_number = input("Which height is required (Hx)??: ")
    path = os.getcwd()
    path = str(path+"/H"+str(set_number)+"/")
    number_files = find_number_files(path)
    #Final array is a 3d array with (points, data, time steps). The data is as follows; x,y,u,v,vel magnitude, vel degree
    data = read_csv(number_files, set_number, path)
    plot(data)
    # print(data.shape)


if __name__=="__main__":
    main()

