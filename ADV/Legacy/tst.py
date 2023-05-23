import copy
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN

# Constants for setting up the particles and their concnetrations
mean = 0.0  # Mean value of the Gaussian distribution
std_dev = 5  # Standard deviation of the Gaussian distribution
pos_start = 0
pos_end = 2
start_index = 10
end_index = 50

# Constants for the simulation
dp  = 0.01
dt = 0.01
time = 10
v = 0.01 # velocity of the fluid
kernelSize = dp*1.4
h = kernelSize


size = int((pos_end-pos_start)/dp) + 1
cOrig = np.zeros(size)
pos = np.linspace(pos_start,pos_end,size)
pp = copy.deepcopy(pos)
gaussian_values = np.zeros(end_index - start_index + 1)


# making the gaussian distribution
for i in range(len(gaussian_values)):
    gaussian_values[i] = abs((1/(std_dev*np.sqrt(2*np.pi)))*np.exp(-(((i-len(gaussian_values)/2)-mean)**2)/(2*std_dev**2)))

# normalizing the gaussian distribution
gaussian_values = gaussian_values/np.max(gaussian_values)
cOrig[start_index:end_index + 1] = gaussian_values
c = cOrig
n= int(time/dt)
time = np.linspace(0,time,n)
c_time = np.zeros([len(c),n])
# SHOWING GRAPHS
scatter = False
line = False

# vol_1 = math.pi*dp*dp/4
volume = 1
vol_1 = 1
vol_0 = 1

def dKer(r):
    q=r/h
    if(q<2):

        #y=-mx
        dk = 1
        #*********
        # qq = 1-q/2
        # dk = qq*qq *q *(-15)/(8*h*h)
        #*********
        # dk = -2*q*math.exp(-q*q)*(1/(np.pi*h*h))
        #*********
        # dk = (1/h*h*math.pi)*(-2*q*math.exp(-q*q))/h
    else:
        dk = 0
    # dk = (1/h*h*math.pi)*(-2*q*math.exp(-q*q))/h
    return dk

# Nearest Neighbors
kdt = NN(radius=kernelSize, algorithm='kd_tree').fit(pp.reshape(1,-1).T)
NN_idx = kdt.radius_neighbors(pp.reshape(1,-1).T)[1]

# iterating through time
for i in range(n):
    ctemp = np.zeros(len(cOrig))
    
    # for first and the last particles
    # ctemp[0] = ((ctemp[0]*vol_0) + (dt * c[0] * (dk) * (-v) *vol_1))/volume
    # ctemp[-1] = ((ctemp[-1]*vol_0) + (dt * c[-2] * (dk) * (v)*vol_1))/volume
    for k in range(1,len(cOrig)-1):
        for j in (NN_idx[k]):
            dk = dKer(abs(pos[k]-pos[j]))
            a = ctemp[j]*vol_0
            b1 = c[k] - c[j]
            # if(b1<10e-10):
            #     continue
            b2 = dk
            b3 = v
            b4 = vol_1
            b5 = pos[k]-pos[j]
            if(b5==0 ):
                continue
            # b = dt * b1 * b2 * b3 * b4/b5
            # b = dt * b1 * b2 * b3 * b4/(pos[k]-pos[j])
            b = dt * (c[k] - c[j]) * (dk) * (v)*vol_1/(pos[k]-pos[j])
            
            ctemp[j] = a + b
            # ctemp[j] = ((ctemp[j]*vol_0) + (dt * (c[j-1] - c[j]) * (dk) * (v)*vol_1/(pos[j]-pos[j-1])))/volume
        # ctemp[j] = ((ctemp[j]*vol_0) + (dt * (c[j] * (dk) * (-v)*vol_1/r))/volume
    # ctemp = ctemp/volume
    # ctemp[0] = (ctemp[0]*vol_0) + (dt * c[0] * (dk) * (-v) *vol_1)
    # ctemp[-1] = (ctemp[-1]*vol_0) + (dt * c[-2] * (dk) * (v)*vol_1)
    # for j in range(1,len(cOrig)-1):
    #     ctemp[j] = (ctemp[j]*vol_0) + (dt * c[j-1] * (dk) * (v)*vol_1)
    #     ctemp[j] = (ctemp[j]*vol_0) + (dt * c[j] * (dk) * (-v)*vol_1)
    # ctemp = ctemp/volume
    c = c - (ctemp)
    c_time [:,i] = c

    # print("total: ", np.sum(c))
    if scatter:
        plt.clf()
        plt.scatter(pos, np.zeros(len(c)),c=c_time[:,i])
        plt.clim(0,1)
        plt.colorbar()
        plt.grid(which='both')
        plt.draw()
        plt.pause(interval=0.01)
    if line:
        plt.clf()
        plt.plot(pos, c_time[:,i])
        plt.xlabel(xlabel="Particle")
        plt.ylabel(ylabel="Concentration")
        plt.title(label="Time: "+str(time[i]))
        plt.grid()
        plt.ylim(0,1)
        plt.draw()
        plt.pause(interval=0.01)

# for i in range(len(c)):
#     plt.plot(time, c_time[i,:], label="c_"+str(i))
print("total at start: ", np.sum(cOrig))
print("position of peak: ", pos[np.argmax(cOrig)])
print("total end: ", np.sum(c_time[:,-1]))
print("position of peak: ", pos[np.argmax(c_time[:,-1])])
plt.plot(pos, cOrig, label="t=0")
plt.plot(pos, c_time[:,-1], label="t=last")
plt.grid(which='both')
plt.xlabel("Pos (m)")
plt.ylabel("Concentration")
plt.legend()
# plt.show()
plt.savefig("concentration.png")
