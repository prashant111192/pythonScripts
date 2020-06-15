#!/usr/bin/python3

import sys
import numpy as np
import csv
import os
# import pandas
import statistics as stat
import matplotlib.pyplot as plt
# from tqdm.auto import tqdm

n = int(input('Number of Timesteps: '))
ts = float(input('Duration of Timesteps: '))
# name = str(input("name of the file"))

def distance_pb(end,start):
    return (10-end+start)


temp = np.genfromtxt('CsvPipe_0000.csv', skip_header=4, delimiter=';', usecols=(0,1,2,3,4,5,6,7,8), dtype=np.float)
comp_arr = np.empty((temp.shape[0],temp.shape[1],n))
comp_arr[:,:,0] = sorted(temp, key=lambda x:x[3])
addition = np.zeros((temp.shape[0]))
# print(temp2)
# print(comp_arr.shape)
temp_pos_pre = comp_arr[:,0,0]

for j in (range(1,n)):
    i=format(int(j),'0>4')
    temp = np.genfromtxt('CsvPipe'+'_'+str(i)+'.csv', skip_header=4, delimiter=';', usecols=(0,1,2,3,4,5,6,7,8), dtype=np.float)
    temp = sorted(temp, key=lambda x:x[3])
    diff_pos = temp[:,0] - temp_pos_pre[:,0] 
    
    for(row in diff_pos):
        if diff_pos[row] < -8:
            comp_arr[row,0,j] = comp_arr[row,0,j-1] + distance(temp_pos_pre[row],temp[row])

        elif diff_pos[row] > 8:
            comp_arr[row,0,j] = comp_arr[row,0,j-1] - distance(temp_pos_pre[row],temp[row])

        else:
            comp_arr[row,0,j] = comp_arr[row,0,j-1] + diff_pos[row]


    comp_arr[:,1:,j]=temp[:,1:]



        













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



