#!/usr/bin/python3

import sys
import numpy as np
import csv
import os
import time
# import pandas
import statistics as stat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from tqdm.auto import tqdm

n = int(input('Number of Timesteps: '))
ts = float(input('Duration of Timesteps: '))
# name = str(input("name of the file"))

timesteps = np.arange(n)
timesteps = ts*timesteps
def distance_pb(end,start):
    return (10-end+start)


temp = np.genfromtxt('CsvPipe_0000.csv', skip_header=4, delimiter=';', usecols=(0,1,2,3,4,5,6,7,8), dtype=np.float)
comp_arr = np.empty((temp.shape[0],temp.shape[1],n))
comp_arr[:,:,0] = sorted(temp, key=lambda x:x[3])
addition = np.zeros((temp.shape[0]))
# print(temp2)
# print(comp_arr.shape)
temp_pos_pre = comp_arr[:,0,0]
# print(temp_pos_pre.shape)
# c=0
# d=0
for j in (range(1,n)):
    i=format(int(j),'0>4')
    temp = np.genfromtxt('CsvPipe'+'_'+str(i)+'.csv', skip_header=4, delimiter=';', usecols=(0,1,2,3,4,5,6,7,8), dtype=np.float)
    # temp = sorted(temp, key=lambda x:x[3])
    # print(temp.shape)
    # temp = np.sort(temp, order =['f0'], axis =0)
    temp.view('i8,i8,i8,i8,i8,i8,i8,i8,i8').sort(order=['f3'],axis = 0)  
    # print(temp.shape)
    # print(type(temp))
    diff_pos = temp[:,0] - temp_pos_pre[:] 
    # print(diff_pos)
    
    for k in range(len(diff_pos)):
        if diff_pos[k] < -8:
            comp_arr[k,0,j] = comp_arr[k,0,j-1] - distance_pb(temp_pos_pre[k],temp[k,0])
            # temp[k,0] = comp_arr[k,0,j-1] + distance_pb(temp_pos_pre[k],temp[k,0])
            # print(1)

        elif diff_pos[k] > 8:
            comp_arr[k,0,j] = comp_arr[k,0,j-1] + distance_pb(temp_pos_pre[k],temp[k,0])
            # temp[k,0] = comp_arr[k,0,j-1] - distance_pb(temp_pos_pre[k],temp[k,0])
            # c=c+1
            # print(2)

        else:
            # temp[k,0] = comp_arr[k,0,j-1] + diff_pos[k]
            comp_arr[k,0,j] = comp_arr[k,0,j-1] + diff_pos[k]
            # d=d+1
            # print(3)
    comp_arr[:,1:,j]=temp[:,1:]
    temp_pos_pre = temp[:,0]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')


# plt.ion()
# fig=plt.figure()
# ax = Axes3D(fig) 
# ax = fig.add_subplot(111, projection='3d')
# i=0
# for i in range(n):
    # plt.cla()
    # ax.scatter(comp_arr[:,0,i], comp_arr[:,1,i], comp_arr[:,2,i])
    # i=i+1
    # plt.draw()
    # plt.pause(0.5)
    # fig.clear()
# plt.pause(0.5)
# plt.clf()
#
# fig.clf()
# plt.show()

# print(comp_arr.shape) 

# fig,  ax = plt.subplots()
# ax.plt(comp_arr[1,0,:],time)
# plt.plot(comp_arr[120,0,:],timesteps)
fig=plt.figure()
# ax = Axes3D(fig) 
ax = fig.add_subplot(111, projection='3d')
ax.scatter(comp_arr[:,0,1999], comp_arr[:,1,1999], comp_arr[:,2,1999])
plt.xlabel('x')
plt.ylabel('y')
# plt.zlabel('z')
plt.show()

    # print(comp_arr.shape)


# print(c)
# print(d)

    # print(type(i))
    # i=format(int(i),'0>4')
    # print(i)
    # print(type(i))
    # csvarr = np.genfromtxt('CsvPipe'+'_'+str(i)+'.csv', skip_header=4, delimiter=';', usecols=(0,1,2,3,4,5,6,7,8), dtype=np.float)
    # with open('CsvPipe_'+str(i)+'.csv') as csv_file:
        # csvarr = csv.reader(csv_file, delimiter=';') 
        # print(type(csvarr))
        # lineCount = 0 
        # csvarr = list(csvarr)
    # csvarr = np.array(csvarr)
    # print(csvarr[1])
    # print(csvarr.shape)
        # print(np.array(csvarr[1]))
        # print(type(csvarr))
        # print(type(data))
        # print(csvarr)

        # for row in csvarr[3,:]:
            # print(','.join(row))
            # if lineCount > 5:
                # print(type(csv_reader[:,row]))

        #     lineCount +=1
        #     print(lineCount)
# print(csv_reader)
                # print(f'{",".join(row)}')
            # if lineCount > 5:
                # a=np.append(a,csv_reader(row))

        # print(type(csv_reader)
    # temp = np.loadtxt('CsvPipe'+'_'+str(i)+'.csv', delimiter = ';', skiprows =4)
    # temp = np.loadtxt('CsvPipe_'+str(i)+'.csv', delimiter = ';', skiprows =4)
    # print(type(i))
    # sorted(temp, 3) 
    # temp[:,[0,3]] = temp[:,[3,0]]
    # np.append(files,np.atleast_3d(temp))

# print(files)
# print(files.shape)

# add = 0

# for i in tqdm(range(1,n)):
    # if     



