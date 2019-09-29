import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

fig, ax = plt.subplots()
plt.xlabel('epochs')
plt.ylabel('validation error')


data_ = np.array((
    [np.loadtxt('data_as_it_is.txt'), ('Original Model')],
))

