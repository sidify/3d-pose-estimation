import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.loadtxt('data')
x = np.array((x))
y = range(0, x.size)
x_ = np.loadtxt('data_no')
y_ = range(0, x_.size)
plt.xlabel('epochs')
plt.ylabel('validation error')
plt.plot(y,x, label='No Augmentation')
plt.plot(y_,x_, label='Augmentation with random rotation' )
plt.gca().legend(('Training with noisy data', 'No Augmentation'))
plt.show()