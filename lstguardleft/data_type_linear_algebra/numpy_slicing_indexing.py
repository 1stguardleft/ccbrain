# Basic Slicing and Indexing

import numpy as np

print('[1]--------------------------------------------')

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x[1:7:2])

print('\n[2]--------------------------------------------')

print(x[-2:10])
print(x[-3:3:-1])
