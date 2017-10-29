import numpy as np
import matplotlib.pyplot as plt

array = np.random.random([100,4])-0.5
#print('print array')
#print(array)
U,A,V=np.linalg.svd(array,full_matrices=True)
print(A)

array[:,1]=array[:,0]
array[:,2]=array[:,0]
array[:,3]=array[:,0]
plt.scatter(array[:,0],array[:,1])
plt.show()
