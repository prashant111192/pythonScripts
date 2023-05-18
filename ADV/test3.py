import numpy as np
import matplotlib.pyplot as plt

mean = 0.0  # Mean value of the Gaussian distribution
std_dev = 5  # Standard deviation of the Gaussian distribution
size = 100  # Size of the array

cOrig = np.zeros(size)
start_index = 10
end_index = 50

gaussian_values = np.zeros(end_index - start_index + 1)
for i in range(len(gaussian_values)):
    print(i)
    gaussian_values[i] = abs((1/(std_dev*np.sqrt(2*np.pi)))*np.exp(-(((i-len(gaussian_values)/2)-mean)**2)/(2*std_dev**2)))

gaussian_values = gaussian_values/np.sum(gaussian_values)
print(len(gaussian_values))
print(gaussian_values)
cOrig[start_index:end_index + 1] = gaussian_values

plt.plot(cOrig)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bell Curve Plot')
plt.show()
