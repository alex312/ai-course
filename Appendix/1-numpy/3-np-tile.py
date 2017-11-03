#!/usr/bin/env python
#pylint: disable=C0103

import numpy as np

a = np.array([0,1,2])
print(np.tile(a,2))
print()
print(a)
print()
print(np.tile(a,(1,2)))
print()
print(np.tile(a,(3,2)))
print()
print(np.tile(a,(1,3,2)))

print("-"*30)
b = np.tile(a,(2,1,2))
print(b)
print("-"*30)
print(np.tile(b,(1,2)))
print("-"*30)
print(np.tile(b,(2,1,2)))