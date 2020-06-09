#!/usr/bin/python3

import sys
import numpy as np
import csv
import os
import sys
import pandas
import statistics as stat
import matplotlib.pyplot as plt


files=[]
i=0
n = int(input('Number of csv files: '))
skip = int(input('Last how many steps: '))


for i in range(n):
    a = np.loadtxt('GaugesVel_Velpt'+str(i)+'.csv', delimiter=';',skiprows=1)
    files.append(a)
    #print (a)
    #vavg[i] = stat.mean(int(a[:,3]))

    #print (a.shape)
files=np.array(files)


len_all = files.shape
ts = len_all[1]-1
skip = ts-skip
vavg = []
vavg = np.array(vavg)

vlast = []
vlast = np.array(vlast)

time = []

xloc = []
xloc = np.array(xloc)

zloc = []
zloc = np.array(zloc)

i = skip 
time = files[1,skip:,0]


for i in range (len_all[0]):
    tmpavg = stat.mean(files[i,skip:,3])
    vavg = np.append(vavg, tmpavg)
    vlast =np.append(vlast, files[i,ts,3])
    xloc = np.append(xloc, files[i,0,4])
    zloc = np.append(zloc, files[i,0,6])

    #vavg[i] = stat.mean(files[i,:,3])
    #vlast[i] = files[i,int(len_all[1])-1,3]
    #xloc[i] = files[i,0,4]
    #zloc[i] = files[i,0,6]


plt.plot(xloc,vavg)
plt.title("Axial velocity at 6m from digester bottom")
plt.legend("velZ (m/s)")
plt.xlabel("radial position")
plt.ylable("velocity m/s")
plt.grid(True)
plt.show()

#print (vavg)

#print (vlast)

