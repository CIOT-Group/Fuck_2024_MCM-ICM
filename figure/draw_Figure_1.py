import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

plt.xlim(-0.1, 4)
plt.ylim(-0.2, 1.2)

plt.axhline(y=1, color='#A6A6A6', linewidth=0.5)
plt.axhline(y=0.8, color='#A6A6A6', linewidth=0.5)
plt.axhline(y=0.6, color='#A6A6A6', linewidth=0.5)
plt.axhline(y=0.4, color='#A6A6A6', linewidth=0.5)
plt.axhline(y=0.2, color='#A6A6A6', linewidth=0.5)


u = 0.7
sig = math.sqrt(0.05)
x = np.linspace(-2, 2, 200)
y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / \
    (math.sqrt(2*math.pi)*sig)
plt.plot(y, x, color='black', linewidth=2)

i = np.linspace(-5, 5, 200)
j = 0.5 * (1 + erf((i - 2) / (0.5 * 1.414)))
plt.plot(i, j, color='#ED7D31', linewidth=2)

plt.axhline(y=0, color='black', linewidth=1)
plt.axvline(x=0, color='black', linewidth=1)

plt.axis('off')
plt.show()
