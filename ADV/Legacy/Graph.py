import matplotlib.pyplot as plt
import numpy as np
c0Orig = 0
c1Orig = 1
c2Orig = 1
c3Orig = 0
c0 = c0Orig
c1 = c1Orig
c2 = c2Orig
c3 = c3Orig
dt = 0.01
time = 1
n= int(time/dt)
time = np.linspace(0,time,n)
c = np.zeros([4,n])
for i in range(n):
    c0temp = 0
    c1temp = 0
    c1temp = 0
    c2temp = 0
    c3temp = 0
    c0temp = c0temp + dt * c0 * (-1) * (-1)
    c1temp = c1temp + dt * c0 * (-1) * (1)
    c1temp = c1temp + dt * c1 * (-1) * (-1)
    c2temp = c2temp + dt * c1 * (-1) * (1)
    c2temp = c2temp + dt * c2 * (-1) * (-1)
    c3temp = c2temp + dt * c2 * (-1) * (1)
    c0 = c0 -c0temp
    c1 = c1 - c1temp
    c2 = c2 - c2temp
    c3 = c3 - c3temp
    c[0,i] = c0
    c[1,i] = c1
    c[2,i] = c2
    c[3,i] = c3
    print("total: ", c0+c1+c2+c3)

plt.plot(time, c[0,:], label="c0")
plt.plot(time, c[1,:], label="c1")
plt.plot(time, c[2,:], ".",label="c2")
plt.plot(time, c[3,:], ".",label="c3")
plt.xlabel("time (s)")
plt.ylabel("Concentration")
plt.legend()
plt.show()
