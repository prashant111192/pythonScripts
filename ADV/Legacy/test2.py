import matplotlib.pyplot as plt
import numpy as np

cOrig = np.array([0, 1, 1, 0])  # Initial concentrations: c0Orig, c1Orig, c2Orig, c3Orig

dt = 0.01
time = 1
n = int(time / dt)
time = np.linspace(0, time, n)

c = np.zeros((len(cOrig), n))  # Array to store concentrations over time

for i in range(n):
    ctemp = np.zeros_like(cOrig)  # Temporary array for intermediate calculations
    
    for j in range(len(cOrig)):
        for k in range(j + 1):
            ctemp[j] += dt * c[k, i] * (-1) * ((-1) ** (j - k))
    
    c[:, i] = cOrig - ctemp
    
    print("total: ", np.sum(c[:, i]))

plt.plot(time, c[0, :], label="c0")
plt.plot(time, c[1, :], label="c1")
plt.plot(time, c[2, :], ".", label="c2")
plt.plot(time, c[3, :], ".", label="c3")
plt.xlabel("time (s)")
plt.ylabel("Concentration")
plt.legend()
plt.show()
