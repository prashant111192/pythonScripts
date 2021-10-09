#!/usr/bin/python3

import sys
import numpy as np
import csv
import os
import statistics as stat
import matplotlib.pyplot as plt


# Input sphase and fluent files
sphase = np.loadtxt('./Sphase.csv', delimiter = ',', skiprows =1)
fluent = np.loadtxt('./FluentDigester.csv', delimiter = ',', skiprows =1)

# sphase = np.array(c)
# fluent = np.array(b)


name = ["DSPH.015.015.100.csv", "dsph.csv", "DSPH.015.060.100.csv"]
# name = "dsph.csv"
# name = "DSPH.015.060.100.csv"
l2_sphase = np.ones(len(name))
l2_fluent = np.ones(len(name))
l1_sphase = np.ones(len(name))
l1_fluent = np.ones(len(name))
for i in range (len(name)):
    dsph = np.loadtxt(f"{name[i]}", delimiter = ',', skiprows =1)
    dsph_inter_sphase = np.interp(sphase[:,0], dsph[:,0], dsph[:,1])
    dsph_inter_fluent = np.interp(fluent[:,0], dsph[:,0], dsph[:,1])

    l2_sphase[i] = np.sum(np.power((sphase[:,1]-dsph_inter_sphase)/len(sphase[:,1]),2))
    l2_fluent[i] = np.sum(np.power((fluent[:,1]-dsph_inter_fluent)/len(fluent[:,1]),2))
    l1_sphase[i] = np.sum(np.absolute((sphase[:,1]-dsph_inter_sphase)))/len(sphase[:,1])
    l1_fluent[i] = np.sum(np.absolute((fluent[:,1]-dsph_inter_fluent)))/len(fluent[:,1])

time = [100, 5, 1]
# plt.plot(time, l1_sphase)
# plt.plot(time, l1_fluent)
# plt.show()

sph_fluent = np.interp(fluent[:,0], sphase[:,0], sphase[:,1])
l2_sphase_fluent = np.sum(np.power((fluent[:,1]-sph_fluent)/len(fluent[:,1]),2))
l1_sphase_fluent = np.sum(np.absolute((fluent[:,1]-sph_fluent)))/len(fluent[:,1])
print(l1_sphase_fluent, l2_sphase_fluent)
print("fluent")
print(l1_fluent,l2_fluent)
print("sphase")
print(l1_sphase, l2_sphase)
# total_len = files.shape #for the number of timesteps
# # print(total_len)
# # timestep=total_len[1]-1
# skip = total_len[1]-skip
# # skip = skip-1


# vavg = np.zeros(n)
# vdiv = np.zeros(n)
# xloc = np.zeros(n)

# counter = 0
# numberloc=n



# # for i in (range (skip,total_len[1])):
# for i in (range (numberloc)):
#     tempavgvel = stat.mean(files[counter,skip:,3])
#     tempdivvel = stat.stdev(files[counter,skip:,3])
#     vavg[counter] = tempavgvel
#     vdiv[counter] = tempdivvel
#     xloc[counter] = files[counter,0,4]
#     counter = counter+1

# # xloc=files[1,:]
# # vavg=files[0,:]
# plt.plot(dsph[:,0],dsph[:,1], 'r-x', label='DSPH')
# # plt.plot(sphase[:,0],sphase[:,1], 'r-x', label='DSPH')
# plt.plot(sphase[:,0],yn, 'g-x', label='DSPH')
# plt.ylim(-0.2, 0.05)
# plt.show()
# #plt.plot(xloc,vdiv, 'b')
# plt.plot(sphase[:,0],sphase[:,1], 'r-.', label='SPHASE')
# plt.plot(fluent[:,0],fluent[:,1], 'b', label='Fluent')
# plt.legend(loc='lower right')
# plt.xlabel("Radial Position(m)")
# plt.ylabel("Axsial Velocity (m/s)")
# plt.savefig('graph_'+str(sys.argv[2])+'_'+str(sys.argv[3])+'_'+str(sys.argv[1])+'.eps' ,format='eps')
# plt.savefig('graph_'+str(sys.argv[2])+'_'+str(sys.argv[3])+'_'+str(sys.argv[1])+'.png' ,format='png')
# plt.plot(xloc,vavg, 'g:', label='DSPH')
# np.savetxt("./dsph.csv", np.stack((xloc, vavg), axis=1), fmt="%f", delimiter=",", comments="", header="xloc,vavg")

