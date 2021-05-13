#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import math
from tqdm import tqdm 
from scipy.spatial import cKDTree
from pyevtk.hl import pointsToVTK

# import time 
# import multiprocessing
# from joblib import Parallel, delayed
# from  datetime import datetime
# from sklearn.neighbors import BallTree
# import gc
# import pandas as pd


# read multiple csv
def read_csv(first,last):
    ##0_Pos.x[m];1_Pos.y[m];2_Pos.z[m];3_Idp;4_Vel.x[m/s];5_Vel.y[m/s];6_Vel.z[m/s];7_Rhop[kg/m^3];8_Type;
    #Bigfiles draft and splash
    arr1 = np.loadtxt('draftsplash_000'+str(first)+'.csv', dtype=float, delimiter=';', skiprows=4, usecols=(0,2,3,8))
    arr2 = np.loadtxt('draftsplash_00'+str(last)+'.csv', dtype=float, delimiter=';', skiprows=4, usecols=(0,2,3,8))
    #small files
    # arr1 = np.loadtxt('withoutdraft_1104.csv', dtype=float, delimiter=';', skiprows=4, usecols=(0,2,3,8))
    # arr2 = np.loadtxt('withoutdraft_1105.csv', dtype=float, delimiter=';', skiprows=4, usecols=(0,2,3,8))

    ## picked 0_x,1_z,2_idp and 3_type
    # extract only the fluids
    arr1 = arr1[arr1[:, 3]==3]
    arr1 = np.delete(arr1,3,1)
    arr2 = arr2[arr2[:, 3]==3]
    arr2 = np.delete(arr2,3,1)
    return arr1, arr2

# Sort array based on idp -> easier for calculating
def sort(arr, pos = 2):
    arr = arr[np.argsort(arr[:,pos])]
    return (arr)

def match_idp(arr1, arr2):
    both = set(arr1[:,2]).intersection(arr2[:,2])
    index_1 = [i for i, item in enumerate(arr1[:,2]) if item in both]
    index_2 = [i for i, item in enumerate(arr2[:,2]) if item in both]
    arr1 = arr1[index_2,:]
    arr2 = arr2[index_2,:]
    return arr1, arr2

# adding displacement column
def displacement(arr,i):
    x = arr[:,0]
    z = arr[:,1]
    x = x - arr[i,0]
    z = z - arr[i,1]
    sqrt_vec = np.vectorize(math.sqrt)
    disp = sqrt_vec((x*x) + (z*z))
    disp = disp.reshape(len(disp),1)
    arr_temp = np.append(arr,disp,1)
    return (arr_temp)

#getting the useful array
def get_arr(temp_indices,arr):
    #array to be built
    temp_arr = arr[temp_indices]
    return temp_arr

#max ratio and sqrt with ln 
def ratio(arr1,arr2):
    inv_arr1 = 1/arr1[:,3]
    temp_div=np.outer(arr2[:,3],inv_arr1)
    ind = np.unravel_index(np.argmax(temp_div, axis=None), temp_div.shape)
    sig_T = np.log(np.sqrt(temp_div[ind]))
    return sig_T

def cKDTree_method(arr1,h):
    tree =cKDTree(arr1[:,0:2])
    nn_dist,index= tree.query(arr1[:,0:2],k=20, distance_upper_bound=(2*h), workers=12)
    # nn_dist, index = dists[0][:,1]
    # tree = BallTree(arr1[:,0:2], leaf_size=50)  
    # nn_dist, index = tree.query_radius(arr1[:,0:2], r=2*h)
    return (nn_dist, index)

def find_distance(arr, x1, z1):
    x = arr[:,0]
    z = arr[:,1]
    x = x - x1
    z = z - z1
    sqrt_vec = np.vectorize(math.sqrt)
    disp = sqrt_vec((x*x) + (z*z))
    return (disp)

def multi_fx_kd(count, m_index, m_dist, arr1, arr2):
    temp_index = m_index[count,1:].compressed()
    #index based on nearest neighbour
    temp_dist1 = m_dist[count,1:].compressed()
    #particles to be chosen in arr2. the are in the same aroder as the distance
    chosen_particles_2 = arr2[temp_index,:]
    temp_dist2 = find_distance(chosen_particles_2,arr2[count,0],arr2[count,1])
    # choosing non zero particle distances
    # non_zero = temp_dist1!=0
    # temp_dist1 = temp_dist1[non_zero]
    # temp_dist2 = temp_dist2[non_zero]
    # non_zero = temp_dist2!=0
    # temp_dist1 = temp_dist1[non_zero]
    # temp_dist2 = temp_dist2[non_zero]

    # remove extreme cases
    ex = temp_dist1<15
    temp_dist1 = temp_dist1[ex]
    temp_dist2 = temp_dist2[ex]
    ex = temp_dist2<15
    temp_dist1 = temp_dist1[ex]
    temp_dist2 = temp_dist2[ex]
    if np.size(temp_dist1):
        i = np.min(temp_dist1)
        j = np.min(temp_dist2)
        ratio = np.max(temp_dist2/temp_dist1)
    else:
        # print ('test')
        ratio = 1
        i=0
        j=0
    return ratio

def to_vtk(sig):
    pointsToVTK('./ftle_vtk',sig[:,0],sig[:,1], data={"ftle": sig[:,2]})



def plot(arr1,sig):
    # plt.tricontour(arr1[:, 0], arr1[:, 1], sig[:, 0], 2, linewidths=0.5)
    marker_size = 1
    plt.scatter(arr1[:,0],arr1[:,1], marker_size,c=sig[:,2])
    plt.colorbar()
    plt.clim(0,1.5)
    plt.gca().set_aspect('equal')
    plt.savefig('ftle.png', bbox_inches='tight')
    # plt.show()

def main():
    # first = int(input('Enter first file number = '))
    # last = int(input('Enter last file number = '))
    first = 0
    last = 10
    # T= 0.015
    T= 5
    h = 0.007071

    arr1, arr2 = read_csv(first,last)

    print(np.shape(arr1))
    print(np.shape(arr2))

    # sort on the basis of idp
    arr1 = sort(arr1)
    arr2 = sort(arr2)

    #pick the mathcing ones
    arr1, arr2 =match_idp(arr1, arr2)
    print(np.shape(arr1))
    print(np.shape(arr2))

    (nn_dist,index) = cKDTree_method(arr1,h)
    knn_index = np.isinf(nn_dist)
    m_index = ma.masked_array(index, mask=knn_index)
    m_dist = ma.masked_array(nn_dist, mask=knn_index)

    # arr2 = arr2[0:len(arr2):2,:]
    # print(np.shape(arr2[:,0:1]))
    # np.savetxt("arr2.csv",arr2, delimiter=",")
    # print(np.max(arr1[:,0]))
    # print(np.shape(arr1))

    # print(knn_index)
    # print(m_dist[20:21,:])
    # print(nn_dist[20:21,:])
    # print(m_index[20:21,:])
    # print(index[20:21,:])
    # m_dist = np.delete(m_dist,0,1)
    
    ####
    count=0
    sig = np.zeros([len(arr1),3])
    for i in tqdm(range(len(arr1))):
        # print (count)
        sig[count,2] = multi_fx_kd(count, m_index, m_dist, arr1, arr2)
        sig[count,0] = arr1[count,0]
        sig[count,1] = arr1[count,1]
        count = count+1
    sig[:,2] = (np.log(np.sqrt(sig[:,2]))/T)
    # num_cores = multiprocessing.cpu_count()-2
    # sig = Parallel(n_jobs = num_cores)(delayed(multi_fx_kd)(alpha,m_index,m_dist,arr1,arr2) for alpha in tqdm(range(count)))
    plot(arr1,sig)
    to_vtk(sig)
    ####
    # print(m_index[21,:])
    # print(m_index[20,:])

    # index = index[knn_index]
    # print(m_index)
    # for i in m_index:
    #     print(nn_dist[count,i])
    #     count +=count

    # print(index)
    # print(nn_dist)
    # sig=fnn(arr1,arr2,0.042426,0.30143)
    # np.savetxt("sig.csv",m_index, delimiter=",")
    np.savetxt("sig.csv",sig, delimiter=",")

    # np.savetxt("ckdtree.csv",nn_dist, delimiter=",")
    # np.savetxt("ckdtreeindex.csv",index, fmt='%d',delimiter=",")
    # print(np.shape(sig))
    # # Velocity magnitude
    # arr1 = displacement(arr1)
    # arr2 = displacement(arr2)

    # #Dead volume
    # print('Dead Volume : '+str(amount_of(arr2, 0.001)))
    # print('LVZ : '+str(amount_of(arr2, 0.01)))
    # print('done')
    # print('Velocity gradient (mean) : '+str(vel_grad_(arr1, arr2)))
    # print(arr1[2:5,:])


if __name__=="__main__":
    main()


