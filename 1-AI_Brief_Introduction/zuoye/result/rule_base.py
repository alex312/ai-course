#!/usr/bin/evn python
# pylint disable=C0103

"""
基于规则实现的fizzbuzz
"""

for i in range(1, 101):
    if i % 3 == 0:
        print("fizz")
    elif i % 5 == 0:
        print("buzz")
    elif i % 15 == 0:
        print("fizzbuzz")
    else:
        print(i)