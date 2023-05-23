
import copy
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN

def kernel(r,h):
    q = r/h
    if(q<2):
        qq = 1-q/2
        qr = qq*qq*qq*qq *(q+q+1)*(7/(4*np.pi*h*h))
        return qr
    else:
        return 0

def dkernelG(r,h, dim):
    q = r/h
    if(q<2):
        qr = -2*q*math.exp(-q*q)*(1/(np.pi*h*h))
        if (dim==2):
            return qr/h
        if dim==1:
            return qr


def dkernelW(r,h, dim):
    q = r/h
    if (dim==2):
        if(q<2):
            qq = 1-q/2
            qr = qq*qq*qq *(-5*q)*(7/(4*np.pi*h*h*h))
            return qr
        else:
            return 0
    else: 
        if (dim==1):
            if(q<2):
                qq = 1-q/2
                qr = qq*qq *q *(-15)/(8*h*h)
                return qr
            else:
                return 0

def main():
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
    kernelType = "Gaussian"
    # kernelType = "Wendland"
    # kernelType = "Flat"
    dim = 1


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
    # creating the vector for original concentrations before any calcs
    cOrig[start_index:end_index + 1] = gaussian_values
    c = cOrig
    n= int(time/dt)
    time = np.linspace(0,time,n)
    c_time = np.zeros([len(c),n])
    # SHOWING GRAPHS
    scatter = False
    line = False


    # vol_1 = math.pi*dp*dp/11
    volume = 1
    vol_1 = 1
    vol_0 = 1

            
    # Nearest Neighbors
    kdt = NN(radius=kernelSize, algorithm='kd_tree').fit(pp.reshape(1,-1).T)
    NN_idx = kdt.radius_neighbors(pp.reshape(1,-1).T)[1]

    # iterating through time
    for i in range(n):
        ctemp = np.zeros(len(cOrig))
        for k in range(1,len(cOrig)-1):
            for j in (NN_idx[k]):
                r= np.norm(pp[k]-pp[j])
                if(kernelType=="Gaussian"):
                    dker = dkernelG(r,h, dim)
                if(kernelType=="Wendland"):
                    dker = dkernelW(r,h, dim)
                if(kernelType=="Flat"):
                    dker = -1
                if (dker<0):
                    dker = 0
                temp = 
                

                

if __name__ == '__main__':
    main()