#!/usr/bin/python3

import sys
import numpy as np
import csv
import os
import statistics as stat
import matplotlib.pyplot as plt

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

# n = int(input('Number of CSV files: '))
# skip = int(input('Last how many steps: '))
# the number of csv is fixed and the skip is taken from the arguments
n=99
skip=int(sys.argv[1])

# Input sphase and fluent files
c = np.loadtxt('./../../../Sphase.csv', delimiter = ', ', skiprows =0)
b = np.loadtxt('./../../../FluentDigester.csv', delimiter = ', ', skiprows =0)

sphase = np.array(c)
fluent = np.array(b)

# plt.plot(sphase[:,0],sphase[:,1], 'r.')
# plt.plot(fluent[:,0],fluent[:,1], 'b.')
# plt.savefig('books_read.png')

# a = np.loadtxt('GaugesVel_6Velpt'+str(i)+'.csv', delimiter = ';', skiprows =1)
# for i in tqdm(range(n)):

i = 0
files = []
for i in range(n):
    a = np.loadtxt('GaugesVel_Velpt'+str(i)+'.csv', delimiter = ';', skiprows =1)
    files.append(a)
    # print(len(a))
    # print(len(files))

i=0

files = np.array(files)

total_len = files.shape #for the number of timesteps
# print(total_len)
# timestep=total_len[1]-1
skip = total_len[1]-skip
# skip = skip-1


vavg = np.zeros(n)
vdiv = np.zeros(n)
xloc = np.zeros(n)

counter = 0
numberloc=n



# for i in (range (skip,total_len[1])):
for i in (range (numberloc)):
    tempavgvel = stat.mean(files[counter,skip:,3])
    tempdivvel = stat.stdev(files[counter,skip:,3])
    vavg[counter] = tempavgvel
    vdiv[counter] = tempdivvel
    xloc[counter] = files[counter,0,4]
    counter = counter+1

# xloc=files[1,:]
# vavg=files[0,:]
plt.plot(xloc,vavg, 'g-x', label='DSPH')
#plt.plot(xloc,vdiv, 'b')
plt.plot(sphase[:,0],sphase[:,1], 'r-.', label='SPHASE')
plt.plot(fluent[:,0],fluent[:,1], 'b', label='Fluent')
plt.legend(loc='lower right')
plt.xlabel("Radial Position(m)")
plt.ylabel("Axsial Velocity (m/s)")
plt.ylim(-0.2, 0.05)
plt.savefig('graph_'+str(sys.argv[2])+'_'+str(sys.argv[3])+'_'+str(sys.argv[1])+'.eps' ,format='eps')
plt.savefig('graph_'+str(sys.argv[2])+'_'+str(sys.argv[3])+'_'+str(sys.argv[1])+'.png' ,format='png')
plt.plot(xloc,vavg, 'g:', label='DSPH')
np.savetxt("./dsph.csv", np.stack((xloc, vavg), axis=1), fmt="%f", delimiter=",", comments="", header="xloc,vavg")

