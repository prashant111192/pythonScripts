#!/usr/bin/python3

import sys
import numpy as np
import math

print ('Rotation of points, enter the x & y cordinates and then the angle in degrees')
xo = int (input('x Coordinate: '))
yo = int (input('y Coordinate: '))
deg = int (input('Enter the angle in degrees(Clockwise-ve): '))

xr = 175 #int (input('x center of rotation: '))
yr = 0#int (input('y center of rotation: '))
rad = (deg/180)*math.pi

c_mat = np.array([[math.cos(rad),-math.sin(rad)],[math.sin(rad),math.cos(rad)]])
xo_mat = np.array((xo,yo))
xr_mat = np.array((xr,yr))
xc_mat = xo_mat-xr_mat
x1_mat = np.dot(c_mat,xc_mat)
result = x1_mat+xr_mat
print ('the new coordinates are', result)
