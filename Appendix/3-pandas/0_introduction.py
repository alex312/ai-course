#!/usr/bin/env python
# coding=utf-8
"""
10 分钟了解pandas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
对象创建
"""
# 通过传递一个list创建一个Series，pandas会自动创建默认的整形索引
series = pd.Series([1, 3, 5, np.nan, 6, 8])
print(series)
# 通过传递一个ndarray创建一个Series，pandas会自动创建默认的整形索引
series = pd.Series(np.array([1, 2, 3, 45, 0]))
print(series)

# 通过传递一个ndarray创建创建一个具有日期索引和列名DataFrame
dates = pd.date_range('20170107', periods=6)  # 首先创建一个日期索引
print(dates)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

# 通过传递一个dict对象创建一个DataFrame,字典的每一项value必须可以转换为相同长度的数组
mdict = {
    'A': 1.,
    'B': pd.Timestamp('20170308'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(['test', 'train', 'test', 'train']),
    'F': 'foo'
}

df = pd.DataFrame(mdict)
print(df)
