import numpy as np
import matplotlib.pyplot as plt
# data to plot
n_groups = 15
original_model = (37.59, 43.92, 38.36, 41.81, 44.84, 52.87, 44.80, 39.42, 50.92, 54.41, 42.92, 43.70, 44.86, 33.28, 36.93) #non-sh
our_model = (36.82, 42.33, 39.29, 41.61, 44.07, 51.49, 44.23, 39.30, 51.97, 53.97, 42.32, 42.60, 43.91, 34.72, 37.75)        #sh

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, original_model, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Original Model')

rects2 = plt.bar(index + bar_width, our_model, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Best performing Model: Rotations with 100% Gaussian Noise')


plt.xlabel('Actions')
plt.ylabel('Error in mm')
plt.title('Original Model vs Best Performing Model ( Action-wise performance  )')
plt.xticks(index + bar_width, ('Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing',
                               'Purchases ', 'Sitting ', 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'Walking',
                               'WalkTogether'), rotation='vertical')
plt.legend()
plt.tight_layout()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, '%d' % int(height), ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.show()