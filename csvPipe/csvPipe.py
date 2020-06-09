#!/usr/bin/python3

import sys
import numpy as np
import csv
import os
import pandas
import statistics as stat
import matplotlib.pyplot as plt
# from tqdm.auto import tqdm

n = int(input('Number of Timesteps: '))
# name = str(input("name of the file"))

files = np.atleast_3d(np.empty)

a=[]
for i in (range(n)):
    # print(type(i))
    i=format(int(i),'0>4')
    # print(i)
    # print(type(i))
    with open('CsvPipe_'+str(i)+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';') 
        lineCount = 0 
        for row in csv_reader:
            if lineCount > 5:
                print(type(csv_reader[:,row]))
            lineCount +=1
            print(lineCount)
# print(csv_reader)
                # print(f'{",".join(row)}')
            # if lineCount > 5:
                # a=np.append(a,csv_reader(row))

        # print(type(csv_reader)
    # temp = np.loadtxt('CsvPipe'+'_'+str(i)+'.csv', delimiter = ';', skiprows =4)
    # temp = np.loadtxt('CsvPipe_'+str(i)+'.csv', delimiter = ';', skiprows =100)
    # print(type(i))
    # sorted(temp, 3) 
    # temp[:,[0,3]] = temp[:,[3,0]]
    # np.append(files,np.atleast_3d(temp))

# print(files)
# print(files.shape)

# add = 0

# for i in tqdm(range(1,n)):
    # if     



