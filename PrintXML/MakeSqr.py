#!/usr/bin/python3

import sys
import numpy as np
import math

np.set_printoptions(precision=3)
print ('make a Box at an angle from the lower left corner')

def str2cord (lists):
    # print(lists)
    temp=[]
    temp=lists.split(',')
    return temp

def str2num(lists):
    temp=[]
    i=0
    # print(i)
    # print(lists)
    while i<len(lists):
        # for i in range(0,(len(lists)-1)):
        if lists[i].find('.')!=-1:
            temp.append(float(lists[i]))
            temp.append(format(lists[i], '3'))
        else:
            temp.append(int(lists[i]))
        i=i+1
    return temp

def rotate (point,por):
    c_mat = np.array([[math.cos(rad),-math.sin(rad)],[math.sin(rad),math.cos(rad)]])
    point = np.array(point)
    point = point-por
    point = point.dot(c_mat) 
    point = por+point
    return point

A = str(input('lower left Corner (x,y): '))
print(A)
length = str(input('Lenth in mm: '))
breadth = str(input('Breadth in mm: '))
por = str(input('Point of Rotation(x,y): '))
angle = str(input('Angle from the positive x along the length (Clockwise-negative): '))

# print (A, length, breadth,por,angle)

A = str2cord(A)
por = str2cord(por)

# print (type(angle))
A = str2num(A)
lennum = str2num(length)
brenum = str2num(breadth)
por = str2num(por)
angle = str2num(angle)
# print (por)
# print (type(por[1]))
# print (type(angle[0]))
rad = (angle[0]/180)*math.pi

B = (A[0]+lennum[0], A[1])
C = (A[0]+lennum[0], A[1]+brenum[0])
D = (A[0], A[1]+brenum[0])
por = np.array(por)

#making a rectangle
# 4.............................3
#  |                           |
#  |                           | 
#  |                           |
#  |
#  |
# 1'''''''''''''''''''''''''''''2
A = rotate(A,por)
B = rotate(B,por)
C = rotate(C,por)
D = rotate(D,por)

rect = np.concatenate((A,B,C,D))
print (rect)
print(A,B,C,D)
