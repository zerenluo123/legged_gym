import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])

for i in range(2):
    b = np.array([1, 2, 3, 4])
    a += b  # add rewards at current step
print(a)
