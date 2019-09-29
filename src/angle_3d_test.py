

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def angle_between(v1, v2):
    v1_u = v1 #unit_vector(v1)
    v2_u = v2 #unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / math.pi


x = np.array((
    [2,0,0],
    [3,0,0],
    [3,3,0]
))
#rotate v1
angle = 270 * math.pi / 180
Rz = np.array((
    [math.cos(angle), -math.sin(angle), 0],
    [math.sin(angle), math.cos(angle), 0],
    [0,0,1]
))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0],x[:,1],x[:,2], c='r')
plt.plot(x[:,0],x[:,1],x[:,2], c='b')
print (x)
#rotate
v = x[0] - x[1]
v = np.dot(Rz,v)
v = v + x[1]

x[0] = v
ax.scatter(x[:,0],x[:,1],x[:,2], c='r')
plt.plot(x[:,0],x[:,1],x[:,2], c='y')
print(x)

plt.show()