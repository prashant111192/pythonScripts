#!/usr/bin/python3

import numpy as np
import math
# import pandas as pd


# read multiple csv
def read_csv(start):
    arr1 = np.loadtxt('withoutdraft_'+str(start)+'.csv', dtype=float, delimiter=';', skiprows=3, usecols=(0,1,2,3,4,5,6,7,8))
    arr2 = np.loadtxt('withoutdraft_'+str(start+1)+'.csv', dtype=float, delimiter=';', skiprows=3, usecols=(0,1,2,3,4,5,6,7,8))
    # arr2 = np.loadtxt('withoutdraft_1105.csv', dtype=float, delimiter=';', skiprows=3, usecols=(0,1,2,3,4,5,6,7,8))
    ## extract only the fluids
    arr1 = arr1[arr1[:,8]==3]
    arr2 = arr2[arr2[:,8]==3]
    return arr1, arr2

# Sort array based on idp -> easier for calculating
def sort(arr1, pos = 3):
    arr1 = arr1[np.argsort(arr1[:,pos])]
    return (arr1)


# adding vel magnitude column
def vel(arr1):
    u = arr1[:,4]
    w = arr1[:,6]
    sqrt_vec = np.vectorize(math.sqrt)
    vel_mag = sqrt_vec((u*u) + (w*w))
    vel_mag = vel_mag.reshape(len(vel_mag),1)
    arr1 = np.append(arr1, vel_mag, axis=1)
    return (arr1)

def amount_of(arr, vel):
    condition = arr[:,9]<vel
    total_len = len(arr)
    temp = np.extract(condition, arr)
    extracted_len = len(temp)
    percent = (extracted_len/total_len)*100
    return (percent)

def vel_grad_(arr1, arr2):
    dx = arr2[:,0]-arr1[:,0] 
    du = arr2[:,4]-arr1[:,4] 
    dz = arr2[:,2]-arr1[:,2] 
    dw = arr2[:,6]-arr1[:,6] 
    zero_x = np.where(dx == 0)[0]
    zero_z = np.where(dz == 0)[0]
    zero_ = (np.concatenate((zero_x,zero_z), axis=0))
    dx_temp = np.delete(dx, zero_)
    du_temp = np.delete(du, zero_)
    # print(zero_)
    # tt = np.abs(dx_temp)
    # print(np.min(tt))
    # print(np.max(tt))
    # print(dx_temp.shape)
    # dx_temp = remove_inf(dx)
    # du_x = du/dx_temp
    du_x = du_temp/dx_temp
    dz_temp = np.delete(dz, zero_)
    dw_temp = np.delete(dw, zero_)
    print(np.max(dx_temp))
    print(np.max(du_temp))
    print(np.max(dz_temp))
    print(np.max(dw_temp))
    # dz_temp = remove_inf(dz)
    # dw_z = dw/dz_temp
    dw_z = dw_temp/dz_temp
    print(np.max(du_x))
    print(np.max(dw_z))
    print(np.average(dx_temp))
    print(np.average(du_temp))
    print(np.average(dz_temp))
    print(np.average(dw_temp))
    print(np.average(du_x))
    print(np.average(dw_z))
    sqrt_vec = np.vectorize(math.sqrt)
    vel_grad = np.average((du_x*du_x) + (dw_z*dw_z))
    # vel_grad = sqrt_vec((du_x*du_x) + (dw_z*dw_z))
    # a = np.where(vel_grad>1000)[0]
    # print(a.shape)
    print(np.max(vel_grad))
    avg_grad = math.sqrt(vel_grad)
    # avg_grad = np.average(vel_grad)
    return (avg_grad)

def remove_inf(d):
    zero_ = np.where(d == 0)[0]
    d_temp = np.delete(d, zero_)
    # min_d = np.min(np.abs(d_temp))
    min_d = np.average(np.abs(d_temp))
    print(min_d)
    for i in range(len(zero_)):
        d_temp = np.insert(d_temp,zero_[i], min_d)
        
    return d_temp

def main():
    start = 1104

    arr1, arr2 = read_csv(start)
    # print(arr1[2:5,:])

    # sort on the basis of idp
    arr1 = sort(arr1)
    arr2 = sort(arr2)

    # Velocity magnitude
    arr1 = vel(arr1)
    arr2 = vel(arr2)

    #Dead volume
    print('Dead Volume : '+str(amount_of(arr2, 0.001)))
    print('LVZ : '+str(amount_of(arr2, 0.01)))
    print('Velocity gradient (mean) : '+str(vel_grad_(arr1, arr2)))
    print(arr1[2:5,:])


if __name__=="__main__":
    main()
