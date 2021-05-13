#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import pointsToVTK

def plot(sig):
    # plt.tricontour(arr1[:, 0], arr1[:, 1], sig[:, 0], 2, linewidths=0.5)
    marker_size = 1
    plt.scatter(sig[:,0],sig[:,1], marker_size,c=sig[:,2])
    plt.colorbar()
    plt.clim(0,0.7)
    plt.gca().set_aspect('equal')
    plt.savefig('ftle_testaa.png', bbox_inches='tight')

def to_vtk(arr):
    pointsToVTK("test_ftle", arr[:,0], arr[:,1], np.zeros(len(arr[:,0])), data={"ftle": arr[:,2]})

def main():
    # first = int(input('Enter first file number = '))
    # last = int(input('Enter last file number = '))
    arr = np.loadtxt('ftle_spash.csv', dtype=float, delimiter=',')
    plot(arr)
    # arr = np.array(arr, order='F')
    # to_vtk(arr)


if __name__=="__main__":
    main()


