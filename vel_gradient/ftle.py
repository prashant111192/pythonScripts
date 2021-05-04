#!/usr/bin/python3

import numpy as np
import math
# import time 
import multiprocessing
from joblib import Parallel, delayed
from  datetime import datetime
from tqdm import tqdm 


# import pandas as pd


# read multiple csv
def read_csv(first,last):
    #Preparing first/main array
    #0_Pos.x[m];1_Pos.y[m];2_Pos.z[m];3_Idp;4_Vel.x[m/s];5_Vel.y[m/s];6_Vel.z[m/s];7_Rhop[kg/m^3];8_Type;
    arr1 = np.loadtxt('withdraft_'+str(first)+'.csv', dtype=float, delimiter=';', skiprows=3, usecols=(0,2,3,8))
    arr2 = np.loadtxt('withdraft_'+str(last)+'.csv', dtype=float, delimiter=';', skiprows=3, usecols=(0,2,3,8))
    #picked 0_x,1_z,2_idp and 3_type

    # extract only the fluids
    arr1 = arr1[arr1[:,3]==3]
    # arr1 = arr1[0:10,:]
    arr1 = np.delete(arr1,3,1)
    
    arr2 = arr2[arr2[:,3]==3]
    # arr2 = arr2[0:10,:]
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
    num_cores = multiprocessing.cpu_count()
    sig = Parallel(n_jobs = num_cores)(delayed(multi_fx)(alpha,arr1,arr2,h,T) for alpha in tqdm(count))
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
    # print(np.shape(using_arr1))
    using_arr2 = arr2_temp[temp_indices2]
    # using_arr1 = get_arr(temp_indices,arr1_temp)
    # using_arr2 = get_arr(temp_indices,arr2_temp)
    #find max ratio, sqrt, ln
    sig=ratio(using_arr1,using_arr2)/T
    # print(sig)
    return (sig,count)


def main():
    first = int(input('Enter first file number = '))
    last = int(input('Enter last file number = '))

    arr1, arr2 = read_csv(first,last)
    # print(arr1[2:5,:])
    # print(arr1)

    # sort on the basis of idp
    arr1 = sort(arr1)
    arr2 = sort(arr2)
    # print(np.shape(arr1))

    sig=fnn(arr1,arr2,0.042426,0.30143)
    np.savetxt("sig.csv",sig, delimiter=",")
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


# # adding vel magnitude column
# def vel(arr1):
#     u = arr1[:,4]
#     w = arr1[:,6]
#     sqrt_vec = np.vectorize(math.sqrt)
#     vel_mag = sqrt_vec((u*u) + (w*w))
#     vel_mag = vel_mag.reshape(len(vel_mag),1)
#     arr1 = np.append(arr1, vel_mag, axis=1)
#     return (arr1)

# def amount_of(arr, vel):
#     condition = arr[:,9]<vel
#     total_len = len(arr)
#     temp = np.extract(condition, arr)
#     extracted_len = len(temp)
#     percent = (extracted_len/total_len)*100
#     return (percent)


# def vel_grad_(arr1, arr2):
#     dx = arr2[:,0]-arr1[:,0] 
#     du = arr2[:,4]-arr1[:,4] 
#     dz = arr2[:,2]-arr1[:,2] 
#     dw = arr2[:,6]-arr1[:,6] 
#     zero_x = np.where(dx == 0)[0]
#     zero_z = np.where(dz == 0)[0]
#     zero_ = (np.concatenate((zero_x,zero_z), axis=0))
#     dx_temp = np.delete(dx, zero_)
#     du_temp = np.delete(du, zero_)
#     # print(zero_)
#     # tt = np.abs(dx_temp)
#     # print(np.min(tt))
#     # print(np.max(tt))
#     # print(dx_temp.shape)
#     # dx_temp = remove_inf(dx)
#     # du_x = du/dx_temp
#     du_x = du_temp/dx_temp
#     dz_temp = np.delete(dz, zero_)
#     dw_temp = np.delete(dw, zero_)
#     print(np.max(dx_temp))
#     print(np.max(du_temp))
#     print(np.max(dz_temp))
#     print(np.max(dw_temp))
#     # dz_temp = remove_inf(dz)
#     # dw_z = dw/dz_temp
#     dw_z = dw_temp/dz_temp
#     print(np.max(du_x))
#     print(np.max(dw_z))
#     print(np.average(dx_temp))
#     print(np.average(du_temp))
#     print(np.average(dz_temp))
#     print(np.average(dw_temp))
#     print(np.average(du_x))
#     print(np.average(dw_z))
#     sqrt_vec = np.vectorize(math.sqrt)
#     vel_grad = np.average((du_x*du_x) + (dw_z*dw_z))
#     # vel_grad = sqrt_vec((du_x*du_x) + (dw_z*dw_z))
#     # a = np.where(vel_grad>1000)[0]
#     # print(a.shape)
#     print(np.max(vel_grad))
#     avg_grad = math.sqrt(vel_grad)
#     # avg_grad = np.average(vel_grad)
#     return (avg_grad)
#
# def remove_inf(d):
#     zero_ = np.where(d == 0)[0]
#     d_temp = np.delete(d, zero_)
#     # min_d = np.min(np.abs(d_temp))
#     min_d = np.average(np.abs(d_temp))
#     print(min_d)
#     for i in range(len(zero_)):
#         d_temp = np.insert(d_temp,zero_[i], min_d)
#     return d_temp
