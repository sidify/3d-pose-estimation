
import numpy as np

import cv2
import time

N = 5
b = np.random.randint(-N,N,size=(N,N))
A = (b + b.T)/2
x = np.random.randint(-N,N,size=(1,N))
x = x.T
y = np.zeros(N)
start = time.time()
for j in range(N):
    y[j] = 0
    for k in range(j):
       y[j] = y[j] + A[k,j]*x[k]
    for i in range(j,N):
       y[j] = y[j] + A[j,i]*x[i]

end = time.time()
print(end-start)
#print(y)
start = time.time()
for j in range(N):
    y[j] = 0
    for k in range(N):
       y[j] = y[j] + A[j,k]*x[k]

#print(y)
end = time.time()
print(end-start)


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = np.loadtxt('data')
x = np.array((x))
y = range(0, x.size)
x_ = np.loadtxt('data_no')
print("max: ",np.amax(x_))
print("min: ", np.amin(x_))
y_ = range(0, x_.size)
plt.xlabel('epochs')
plt.ylabel('validation error')
plt.plot(y,x, label='No Augmentation')
plt.plot(y_,x_, label='Augmentation with random rotation' )
plt.gca().legend(('Training with noisy data', 'No Augmentation'))
plt.show()

