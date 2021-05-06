#!/usr/bin/python3

from sklearn.neighbors import BallTree
import gc
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import math
# import time 
import multiprocessing
from joblib import Parallel, delayed
from  datetime import datetime
from tqdm import tqdm 
from scipy.spatial import cKDTree


# import pandas as pd


# read multiple csv
def read_csv(first,last):
    #Preparing first/main array
    #0_Pos.x[m];1_Pos.y[m];2_Pos.z[m];3_Idp;4_Vel.x[m/s];5_Vel.y[m/s];6_Vel.z[m/s];7_Rhop[kg/m^3];8_Type;
    arr1 = np.loadtxt('Points_000'+str(first)+'.csv', dtype=float, delimiter=';', skiprows=4, usecols=(0,2,3,8))
    arr2 = np.loadtxt('Points_000'+str(last)+'.csv', dtype=float, delimiter=';', skiprows=4, usecols=(0,2,3,8))
    #picked 0_x,1_z,2_idp and 3_type
    # arr1 = np.loadtxt('withoutdraft_1104.csv', dtype=float, delimiter=';', skiprows=4, usecols=(0,2,3,8))
    # arr2 = np.loadtxt('withoutdraft_1105.csv', dtype=float, delimiter=';', skiprows=4, usecols=(0,2,3,8))

    # extract only the fluids
    arr1 = arr1[arr1[:,3]==3]
    # arr1 = arr1[0:125,:]
    arr1 = np.delete(arr1,3,1)
    
    arr2 = arr2[arr2[:,3]==3]
    # arr2 = arr2[0:125,:]
    arr2 = np.delete(arr2,3,1)
    # print(arr1)

    return arr1, arr2

# Sort array based on idp -> easier for calculating
def sort(arr1, pos = 2):
    arr1 = arr1[np.argsort(arr1[:,pos])]
    return (arr1)


# adding displacement column
def displacement(arr,i):
    x = arr[:,0]
    # print(x)
    z = arr[:,1]
    x = x - arr[i,0]
    # print(x)
    z = z - arr[i,1]
    # print(z)
    sqrt_vec = np.vectorize(math.sqrt)
    disp = sqrt_vec((x*x) + (z*z))
    disp = disp.reshape(len(disp),1)
    arr_temp = np.append(arr,disp,1)
    # print(np.shape(arr_temp))
    # print(disp-arr_temp[:,3])
    # print(disp)
    # print(arr_temp[:,3])
    # arr_temp[:,8] = disp
    return (arr_temp)

#getting the useful array
def get_arr(temp_indices,arr):
    #array to be built
    temp_arr = arr[temp_indices]
    # print(np.shape(temp_arr))

    return temp_arr

#max ratio and sqrt with ln 
def ratio(arr1,arr2):
    inv_arr1 = 1/arr1[:,3]
    # print(np.shape(inv_arr1))
    temp_div=np.outer(arr2[:,3],inv_arr1)
    # print(np.shape(temp_div))
    ind = np.unravel_index(np.argmax(temp_div, axis=None), temp_div.shape)
    sig_T = np.log(np.sqrt(temp_div[ind]))
    return sig_T

#find nearest neighbours
def fnn(arr1,arr2,h,T):
    #append zero array (in the space used by mk number)
    # arr1[:,3] = np.zeros(len(arr1))
    count = range(len(arr1))
    # num_cores = multiprocessing.cpu_count()
    # sig = Parallel(n_jobs = num_cores)(delayed(multi_fx)(alpha,arr1,arr2,h,T) for alpha in tqdm(count))
    # sig = np.arange(len(arr1))
    # for alpha in tqdm(count):
    #      # print(alpha)
    #      sig[alpha]= multi_fx(alpha,arr1,arr2,h,T)
         # sig = np.concatenate(sig,sig_temp)

    return (sig)


def multi_fx(count,arr1,arr2,h,T):
    arr1_temp = displacement(arr1,count)
    arr2_temp = displacement(arr2,count)
    #choosing indices based on 2h
    temp_indices1 = arr1_temp[:,3]<(2*h)
    # temp_indices - np.argwhere(arr1_temp[:,3]>h,axis = 1)
    # print(temp_indices)
    # print(np.shape(arr1_temp))
    arr1_temp = arr1_temp[temp_indices1]
    arr2_temp = arr2_temp[temp_indices1]
    temp_indices2 = arr1_temp[:,3]!=0
    # arr1_temp = arr1_temp[temp_indices]
    #geting nearest neighbours
    using_arr1 = arr1_temp[temp_indices2]
    print(using_arr1[:,3])
    # print(np.shape(using_arr1))
    using_arr2 = arr2_temp[temp_indices2]
    # using_arr1 = get_arr(temp_indices,arr1_temp)
    # using_arr2 = get_arr(temp_indices,arr2_temp)
    #find max ratio, sqrt, ln
    sig=ratio(using_arr1,using_arr2)/T
    # print(sig)
    return (sig)

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
    # print(arr)
    # print(i)
    z = z - z1
    sqrt_vec = np.vectorize(math.sqrt)
    disp = sqrt_vec((x*x) + (z*z))
    # disp = disp.reshape(len(disp),1)
    # arr_temp = np.append(arr,disp,1)
    return (disp)


def multi_fx_kd(count, m_index, m_dist, arr1, arr2):
    temp_index = m_index[count,1:].compressed()
    # print(temp_index)
    # print(temp_index)
    #index based on nearest neighbour
    temp_dist1 = m_dist[count,1:].compressed()
    # print(temp_dist1)
    # temp_dist2 = 
    #particles to be chosen in arr2. the are in the same aroder as the distance
    chosen_particles_2 = arr2[temp_index,:]
    # print(chosen_particles_2)
    temp_dist2 = find_distance(chosen_particles_2,arr2[count,0],arr2[count,1])

    non_zero = temp_dist1!=0
    temp_dist1 = temp_dist1[non_zero]
    temp_dist2 = temp_dist2[non_zero]
    non_zero = temp_dist2!=0
    temp_dist1 = temp_dist1[non_zero]
    temp_dist2 = temp_dist2[non_zero]


    # remove extreme cases
    ex = temp_dist1<15
    # print(ex)
    temp_dist1 = temp_dist1[ex]
    temp_dist2 = temp_dist2[ex]
    ex = temp_dist2<15
    temp_dist1 = temp_dist1[ex]
    temp_dist2 = temp_dist2[ex]
    # non_zero = np.flatnonzero(temp_dist1)
    # temp_dist2 = temp_dist2[non_zero]
    # non_zero = np.flatnonzero(temp_dist2)
    # temp_dist1 = temp_dist1[non_zero]
    # temp_dist2 = temp_dist2[non_zero]
    # print(temp_index)
    # print(temp_dist1)
    if np.size(temp_dist1):
        i = np.min(temp_dist1)
        j = np.min(temp_dist2)
        # i = (len(temp_dist1)-np.count_nonzero(temp_dist1))
        # j = (len(temp_dist2)-np.count_nonzero(temp_dist2))
        # print(np.count_nonzero(temp_dist2))
        ratio = np.max(temp_dist2/temp_dist1)
    else:
        ratio = 1
        i=0
        j=0
    return ratio, i, j

def plot(arr1,sig):
    marker_size=1
    plt.scatter(arr1[:,0],arr1[:,1], marker_size,c=sig[:,0])
    plt.colorbar()
    plt.savefig('ftle.png')
    # plt.show()



def main():
    # first = int(input('Enter first file number = '))
    # last = int(input('Enter last file number = '))
    first = 0
    last = 5
    # T= 0.015
    T= 2.5
    h = 0.007071

    arr1, arr2 = read_csv(first,last)
    # print(arr1[2:5,:])
    # print(arr1)

    # sort on the basis of idp
    arr1 = sort(arr1)
    # arr1 = arr1[0:len(arr1):2,:]
    # np.savetxt("arr1.csv",arr1, delimiter=",")
    arr2 = sort(arr2)
    print(np.shape(arr1))
    print(np.shape(arr2))
    # arr2 = arr2[0:len(arr2):2,:]
    smallest = min(len(arr1),len(arr2))
    arr1 = arr1[0:smallest,:]
    arr2 = arr2[0:smallest,:]
    print(np.shape(arr1))
    # print(np.shape(arr2[:,0:1]))
    # np.savetxt("arr2.csv",arr2, delimiter=",")
    # print(np.max(arr1[:,0]))
    # print(np.shape(arr1))

    (nn_dist,index) = cKDTree_method(arr1,h)
    knn_index = np.isinf(nn_dist)
    # print(knn_index)
    m_index = ma.masked_array(index, mask=knn_index)
    # m_index = np.delete(m_index,0,1)
    m_dist = ma.masked_array(nn_dist, mask=knn_index)
    # print(m_dist[20:21,:])
    # print(nn_dist[20:21,:])
    # print(m_index[20:21,:])
    # print(index[20:21,:])
    # m_dist = np.delete(m_dist,0,1)
    
    count=0
    sig = np.zeros([len(arr1),3])
    for i in tqdm(range(len(arr1))):
        # print (count)
        sig[count] = multi_fx_kd(count, m_index, m_dist, arr1, arr2)
        count = count+1
    sig[:,0] = (np.log(np.sqrt(sig[:,0]))/T)
    # num_cores = multiprocessing.cpu_count()-2
    # sig = Parallel(n_jobs = num_cores)(delayed(multi_fx_kd)(alpha,m_index,m_dist,arr1,arr2) for alpha in tqdm(range(count)))
    plot(arr1,sig)
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


