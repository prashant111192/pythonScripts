import numpy as np
import time
import sys
import csv
import os
import sys
import pandas
import statistics as stat
import matplotlib.pyplot as plt
from numba import jit, njit, prange

@jit(nopython=True, parallel=True) 
def acqure(n, i):
    
    # for i in prange(n):
    files = np.loadtxt('GaugesVel_Velpt'+str(1)+'.csv', delimiter=';',skiprows=1)
        # files.append(a)
    return files


def main():
    files=[]
    print("aa")
    a = np.loadtxt('GaugesVel_Velpt'+str(0)+'.csv', delimiter=';',skiprows=1)
    files.append(a)
    i=1
    n = int(input('Number of csv files: '))
    a =acqure(n,files,i)
    print(a.shape)


if __name__=='__main__':
    main()
