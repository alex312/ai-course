#!/usr/bin/env python
#pylint: disable=C0103

import numpy as np

arr = np.array([1, 2, 3], ndmin=3)  # 规定创建的数组组最小维度数(秩)是3
print("ndmin = 3:", arr)
arr = np.array([
    [[1], [2], [3]], 
    [[4], [5], [6]]
], ndmin=2)  # 使用一个三维数组初创建，规定创建的数组的最小维度数(秩)是2
print("ndmin = 2:", arr)