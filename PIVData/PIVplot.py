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
        # arr_temp = np.loadtxt(name_file, dtype=float, delimiter=',', skiprows=3, usecols=(0,1,5))
        arr_temp = loadAndVel(name_file)
        arr_temp = arr_temp.reshape(arr_temp.shape[0], arr_temp.shape[1],1)
        # 0x [m],1y [m],2u [m/s],3v [m/s],4vorticity [1/s],5magnitude [m/s],6divergence [1/s],7dcev [1],8simple shear [1/s],9simple strain [1/s],10vector direction [degrees]
        arr = np.append(arr, arr_temp, axis = 2)

    return(arr)

def read_csv_2(path):
    """Read multiple csv files."""
    # 0_Pos.x[m];1_Pos.y[m];2_Pos.z[m];3_Idp;4_Vel.x[m/s];5_Vel.y[m/s];6_Vel.z[m/s];7_Rhop[kg/m^3];8_Type;
    root_dir = path
    arr = np.loadtxt(root_dir+'/csv_0400.csv', dtype=float, delimiter=';', skiprows=5, usecols=(0, 1, 2, 3, 4, 5, 6, 8))
    # arr1 = np.loadtxt(root_dir+'csv_0400.csv', dtype=float, delimiter=';', skiprows=5, usecols=(0, 1, 2, 3, 4, 5, 6, 8))
    # 1Pos.x[m];Pos.y[m];Pos.z[m];Idp;Vel.x[m/s];Vel.y[m/s];Vel.z[m/s];Rhop[kg/m^3];Type;

    # picked 0_x,1_y,2_z, 3_idp 4_vx, 5vy, 6_vz, 7_type
    # extract only the fluids
    low = min(arr[:,0])
    arr = arr[arr[:, 7] == 3]
    # picked 0_x,1_y,2_z, 3_idp 4_vx, 5vy, 6_vz
    arr = np.delete(arr, 7, 1)
    arr = vel(arr)
    # arr2 = arr2[arr2[:, 7] == 3]
    # arr2 = np.delete(arr2, 7, 1)
    return (arr, low)

        
# adding vel magnitude column
def vel(arr):
    u = arr[:,4]
    v = arr[:,6]
    w = arr[:,6]
    sqrt_vec = np.vectorize(math.sqrt)
    vel_mag = sqrt_vec((u*u) + (v*v) + (w*w))
    vel_mag = vel_mag.reshape(len(vel_mag),1)
    arr = np.append(arr, vel_mag, axis=1)
    return (arr)

#reading the csv and computing the directional velocites 
def loadAndVel(name_file):
    arr = np.loadtxt(name_file, dtype=float, delimiter=',', skiprows=3, usecols=(0,1,2,3,5))
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

def closest_point_distance(piv_arr, sph_arr, height):
    height = np.ones(len(piv_arr))*height
    height = height.reshape(len(piv_arr),1)
    # piv_arr = np.squeeze(piv_arr)
    # height = np.tile(height, [])
    # adding heights to piv data i.e., making them 3d
    piv_arr = np.insert(piv_arr,2, height, axis = 1) 
    # piv_arr = np.append(piv_arr, height, axis = 1) 
    closest_index = distance.cdist(piv_arr[:,0:3,0], sph_arr[:, 0:3], metric='euclidean').argmin(1)
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

def subtract_plt(piv_arr,sph_arr, vel_piv_index, vel_sph_index, closest_index_sph):
    sub_array = abs(piv_arr[:, vel_piv_index, 0]-sph_arr[closest_index_sph, vel_sph_index])
    percent_arr = (sub_array*100)/piv_arr[:,vel_piv_index,0]
    return percent_arr, sub_array

def plot_graph(x,y,c_, name, title):
    marker_size = 15
    plt.scatter(x, y, marker_size, c= c_)
    plt.colorbar()
    plt.clim(0,100)
    # plt.gca().set_aspect('equal')
    plt.savefig("./figs/"+title+str(name)+".png")
    plt.clf()
    # plt.show()
    return 

def plot_2_together(piv_arr, sph_arr, closest):
    plt.scatter(sph_arr[:,0], sph_arr[:,1],marker = '.' )
    plt.scatter(sph_arr[closest,0], sph_arr[closest,1], marker= ',')
    plt.scatter(piv_arr[:,0,0], piv_arr[:,1,0])
    # plt.savefig("./fig.png")
    plt.show()
    return
    



def main():

    path = os.getcwd()
    # set_number = input("Which height is required (Hx)??: ")
    set_number = [0,2,3,4,5,6,7,8,9,10,11,12]
    heights_array = [0.105,.115,.125,.150,.160,.170,.200,.210,.220,.245,.255,.265]
    for i in range(12):
        path_file= str(path+"/H"+str(set_number[i])+"/")
        number_files = find_number_files(path_file)
        #Final array is a 3d array with (points, data, time steps). The data is as follows; x,y,u,v,vel magnitude, vel degree
        data = read_csv(number_files, set_number[i], path_file)
        pos_x_piv_min = min(data[:, 0, 0])
        pos_x_piv_max = max(data[:, 0, 0])
        sph_arr, low = read_csv_2(path)
        posy = min(sph_arr[:,1])
        posx = min(sph_arr[:,0])
        height = min(sph_arr[:,2])+ heights_array[i]
        shifted_data= np.copy(data)
        shifted_data[:, 1,:]+=posy*0.9
        shifted_data[:,0] = shifted_data[:,0]-pos_x_piv_min-((pos_x_piv_max - pos_x_piv_min)/2)
        # shifted_data[:, 0]+=posx/2
        closest_index_sph = closest_point_distance(shifted_data, sph_arr, height)
        percent, diff=subtract_plt(shifted_data, sph_arr, 4,7, closest_index_sph)
        # plot_graph(data[:,0,0], data[:,1,0], data[:,4,0])
        plot_2_together(shifted_data, sph_arr, closest_index_sph)
        plot_2_together(data, sph_arr, closest_index_sph)
        # plot_graph(data[:,0,0], data[:,1,0], percent, heights_array[i], "percent")
        print (np.average(percent))
        # plot_graph(data[:,0,0], data[:,1,0], diff, heights_array[i], "difference")
        # print(data.shape)

    return


if __name__=="__main__":
    main()

