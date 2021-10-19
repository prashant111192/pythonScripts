#!/usr/bin/python3

import sys
import numpy as np
import csv
import os
import statistics as stat
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Input sphase and fluent files
sphase = np.loadtxt('./Sphase.csv', delimiter = ',', skiprows =1)
fluent = np.loadtxt('./FluentDigester.csv', delimiter = ',', skiprows =1)

# sphase = np.array(c)
# fluent = np.array(b)


# name = ["DSPH.015.015.100.csv", "DSPH.015.0225.100.csv","dsph.csv","DSPH.015.045.100.csv", "DSPH.015.060.100.csv"]
name = ["DSPH.015.015.100.csv", "DSPH.015.0225.100.csv","dsph.csv","DSPH.015.045.100.csv", "DSPH.015.060.100.csv", "DSPH.015.120.100.csv"]
# name = "dsph.csv"
# name = "DSPH.015.060.100.csv"
l2_sphase = np.ones(len(name))
l2_fluent = np.ones(len(name))
l1_sphase = np.ones(len(name))
l1_fluent = np.ones(len(name))
ar_fluent = np.ones(len(name))
ar_sphase = np.ones(len(name))
pd_sphase = np.ones(len(name))
pd_fluent = np.ones(len(name))


for i in range (len(name)):
    dsph = np.loadtxt(f"{name[i]}", delimiter = ',', skiprows =1)
    dsph_inter_sphase = np.interp(sphase[:,0], dsph[:,0], dsph[:,1])
    dsph_inter_fluent = np.interp(fluent[:,0], dsph[:,0], dsph[:,1])

    l2_sphase[i] = np.power((np.sum(np.power((sphase[:,1]-dsph_inter_sphase),2))),.5)
    l2_fluent[i] = np.power((np.sum(np.power((fluent[:,1]-dsph_inter_fluent),2))),.5)
    # l2_sphase[i] = np.power((np.sum(np.power((sphase[:,1]-dsph_inter_sphase),2)))/len(sphase[:,1]),0.5)
    # l2_fluent[i] = np.power((np.sum(np.power((fluent[:,1]-dsph_inter_fluent),2)))/len(dsph_inter_fluent),.5)
    # l2_sphase[i] = np.power(np.sum(np.power((sphase[:,1]-dsph_inter_sphase),2))/len(sphase[:,1]),0.5)
    # l2_fluent[i] = np.power(np.sum(np.power((fluent[:,1]-dsph_inter_fluent),2))/len(fluent[:,1]),0.5)
    # l1_sphase[i] = np.sum(np.absolute((sphase[:,1]-dsph_inter_sphase)))/len(sphase[:,1])
    # l1_fluent[i] = np.sum(np.absolute((fluent[:,1]-dsph_inter_fluent)))/len(fluent[:,1])
    l1_sphase[i] = np.sum(np.absolute((sphase[:,1]-dsph_inter_sphase)))
    l1_fluent[i] = np.sum(np.absolute((fluent[:,1]-dsph_inter_fluent)))
    temp1 = np.power(dsph[:,1],2)
    temp2 = np.power(sphase[:,1],2)
    ar_sphase[i] = np.power(((np.sum(temp1))/(np.sum(temp2))),.5)
    temp1 = np.power(dsph[:,1],2)
    temp2 = np.power(fluent[:,1],2)
    ar_fluent[i] = np.power(((np.sum(temp1))/(np.sum(temp2))),.5)
    temp1 = (np.sum(np.power(dsph_inter_sphase-sphase[:,1],2)))/(np.sum(np.power(dsph_inter_sphase,2)))
    pd_sphase[i] = np.power(temp1,.5)
    temp1 = (np.sum(np.power(dsph_inter_fluent-fluent[:,1],2)))/(np.sum(np.power(dsph_inter_fluent,2)))
    pd_fluent[i] = np.power(temp1,.5)

# time = [1206, 205, 34]
# time = np.array([1134549, 286430, 73001, 18928])
# time = np.array([0.015, 0.0225, 0.03, 0.045,0.06]).reshape(-1,1)
time = np.array([0.015, 0.0225, 0.03, 0.045,0.06, 0.12]).reshape(-1,1)
time = time*1

errorSphase = l2_sphase
errorFluent = l2_fluent
model = LinearRegression()
model.fit(time, errorFluent)
y_pred = model.predict(time)
# sp = csaps.UnivariateCubicSmoothingSpline(x, y, smooth=0.85)


print(model.coef_)
plt.plot(time, (model.coef_*time+model.intercept_))
plt.scatter(time, errorSphase, label="sphase")
plt.scatter(time, errorFluent, label = "fluent")
axes = plt.gca()
# y_vals = np.array(l2_fluent)
plt.xlim(0.01,1)
x_vals = np.array(axes.get_xlim())
intercept = model.intercept_
# intercept = 0.5*l2_sphase[0]
# intercept = 0
# intercept = -1*np.power(l2_sphase[0],1000000000)
y_vals = (1) + 2 * x_vals
# y_vals = (intercept+.1) + 2 * x_vals
# x_vals = intercept + y_vals/2
plt.plot(x_vals, y_vals, '-.', label = "2nd order convergence")
# x_vals = intercept + y_vals/1
y_vals = (0.1) + 1 * x_vals
plt.plot(x_vals, y_vals, '--', label="1st order convergence")
# y_vals = intercept + 3 * x_vals
# plt.plot(x_vals, y_vals, '--', label="3")
plt.ylim(0.1,10)
plt.xlim(0.01,1)
# plt.ylim(0.5*np.min(errorSphase),2*np.max(errorSphase))
plt.yscale('log')
plt.xscale('log')
plt.grid(which="minor")
plt.grid(which="major")
plt.legend()
plt.savefig("xx.png")
plt.clf()
# time = np.array([1134549, 286430, 73001, 18928])
plt.plot(time, l1_sphase, label="V/s SPHASE")
plt.plot(time, l1_fluent, label = "V/s Fluent")
plt.legend()
plt.yscale('log')
plt.savefig("mesh.png")
# plt.show()

sph_fluent = np.interp(fluent[:,0], sphase[:,0], sphase[:,1])
l2_sphase_fluent = np.sum(np.power((fluent[:,1]-sph_fluent)/len(fluent[:,1]),2))
l1_sphase_fluent = np.sum(np.absolute((fluent[:,1]-sph_fluent)))/len(fluent[:,1])

temp1 = np.power(sph_fluent,2)
temp2 = np.power(fluent[:,1],2)
ar_sf = np.power(((np.sum(temp1))/(np.sum(temp2))),.5)
temp1 = (np.sum(np.power(sph_fluent-fluent[:,1],2)))/(np.sum(np.power(sph_fluent,2)))
pd_sf = np.power(temp1,.5)



print(l1_sphase_fluent, l2_sphase_fluent)
print("fluent")
print(l1_fluent,l2_fluent)
print("sphase")
print(l1_sphase, l2_sphase)
print("pd")
print(pd_sphase)
print(pd_fluent)
print(pd_sf)
print("ar")
print(ar_sphase)
print(ar_fluent)
print(ar_sf)






# plt.plot(dsph[:,0],dsph[:,1], 'r-x', label='DSPH')
# plt.plot(sphase[:,0],sphase[:,1], 'r-x', label='DSPH')
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


# https://www.sciencedirect.com/science/article/pii/S0927025616304426