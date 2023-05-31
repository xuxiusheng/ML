import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    a = []
    for i in range(len(x)):
        a.append(1 / (1 + np.exp(-x[i])))
    return a

x = np.linspace(-10, 10, 1000)

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.patch.set_facecolor('silver')
ax.plot(x, sigmoid(x), label='sigmoid')
plt.grid()
plt.show()