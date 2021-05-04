#!/usr/bin/python3

import numpy as np
import math
# import pandas as pd


# read multiple csv
def read_csv(start):
#/run/user/1000/gvfs/sftp:host=titan-c815/data1/prashant/paper/Massoud2/0.015/0.03/Out2$ 

    arr1 = np.loadtxt('withdraft_'+str(start)+'.csv', dtype=float, delimiter=';', skiprows=3, usecols=(0,1,2,3,4,5,6,7,8))
    arr2 = np.loadtxt('withdraft_'+str(start+1)+'.csv', dtype=float, delimiter=';', skiprows=3, usecols=(0,1,2,3,4,5,6,7,8))
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

#Printing percent of dead and lvz
def amount_of(arr, vel):
    condition = arr[:,9]<vel
    total_len = len(arr)
    temp = np.extract(condition, arr)
    extracted_len = len(temp)
    print (extracted_len)
    print (total_len)
    percent = (extracted_len/total_len)*100
    return (percent)

def vel_grad_(arr1, arr2):
    dx = arr2[:,0]-arr1[:,0] 
    du = arr2[:,4]-arr1[:,4] 
    # tt = abs(dx)
    # print(np.min(tt))
    # du_x = du/dx
    dz = arr2[:,2]-arr1[:,2] 
    dw = arr2[:,6]-arr1[:,6] 
    # dw_z = dw/dz
    # sqrt_vec = np.vectorize(math.sqrt)
    # vel_grad = sqrt_vec((du_x*du_x) + (dw_z*dw_z))
    # avg_grad = np.average(vel_grad)
    return (avg_grad)


def main():
    start = int(input('File Number to start from = '))

    arr1, arr2 = read_csv(start)
    print(arr1[2:5,:])

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
