#!/usr/bin/python3

import sys
import numpy as np
import csv
import os
import pandas
import statistics as stat
import matplotlib.pyplot as plt
# from tqdm.auto import tqdm

n = int(input('Number of CSV files: '))
skip = int(input('Last how many steps: '))

i = 0
files = []

# files = numpy.zeros(shape=(

for i in tqdm(range(n)):
    a = np.loadtxt('GaugesVel_6Velpt'+str(i)+'.csv', delimiter = ';', skiprows =1)
    files.append(a)

files = np.array(files)

total_len = files.shape #for the number of timesteps

skip = skip-1

vavg = numpy.zeroes(shape=(skip))
xloc = numpy.zeroes(shape=(skip))

counter = 0

for i in tqdm(range (skip,total_len[0])):
    tempavgvel = stat.mean(files[i,skip:,3])
    vavg[counter] = [tempavgvel]
    xloc[counter] = [files[i,0,4]]
    counter = counter+1

plt.plot(xloc,vavg)
plt.show

