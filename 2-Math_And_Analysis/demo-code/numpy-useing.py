import numpy as np
import matplotlib.pyplot as plt
a = np.ones([2,2])

print(a)

b =np.random.random()
print(b)

c=np.random.random(1000)
print(c)

pi = np.pi
x = np.linspace(0,2*pi,1000)
y1 = np.sin(x)
y2= np.sin(2*x)
plt.plot()