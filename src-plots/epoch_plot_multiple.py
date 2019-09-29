import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
plt.xlabel('epochs')
plt.ylabel('validation error')


data = np.array((
    [np.loadtxt('data_as_it_is.txt'), ('Original Model')],
    #[np.loadtxt('noise/noise_10p.txt'), ('Gaussian Noise : 10 %')],
    #[np.loadtxt('noise/noise_25p.txt'), ('Gaussian Noise : 25 %')],
    [np.loadtxt('noise/noise_50p.txt'), ('Gaussian Noise : 50 %')],
    #[np.loadtxt('noise/noise_75p.txt'), ('Gaussian Noise : 75 %')],
    #[np.loadtxt('noise/noise_100p.txt'), ('Gaussian Noise : 100 %')] #best
))

data_ = np.array((
    [np.loadtxt('data_as_it_is.txt'),('Original Model')],
    [np.loadtxt('data_flip.txt'),('Augment : Flip')],
    [np.loadtxt('data_rotation.txt'),('Augment : Rotation')], #best
    [np.loadtxt('data_kinematics.txt'),('Augment : Kinematics')]
    ))

data_ = np.array((
    [np.loadtxt('data_as_it_is.txt'),('Original Model')],
    [np.loadtxt('noise/kinematics_noise_50p.txt'),('Augment : Kinematics, Gaussian Noise : 50%')], #best
    [np.loadtxt('noise/kinematics_noise_100p.txt'),('Augment : Kinematics, Gaussian Noise : 100%')]
))

data_ = np.array((
    [np.loadtxt('data_as_it_is.txt'), ('Original Model')],
    [np.loadtxt('noise/rotation_noise_50p.txt'), ('Augment: Rotation, Gaussian Noise : 50%')],
    [np.loadtxt('noise/rotation_noise_100p.txt'),('Augment: Rotation, Gaussian Noise : 100%')], #best
))

data_ = np.array((
    [np.loadtxt('data_as_it_is.txt'),('Original Model')],
    [np.loadtxt('noise/rotation_noise_100p.txt'),('Augment: Rotation, Gaussian Noise : 100%')],
    [np.loadtxt('noise/kinematics_noise_50p.txt'),('Augment : Kinematics, Gaussian Noise : 50%')],
    [np.loadtxt('data_rotation.txt'),('Augment : Rotation')],
    [np.loadtxt('noise/noise_100p.txt'), ('Gaussian Noise : 100 %')]
))

y = range(0, 100)
c = 0
for x, l in data: #zip(data, labels):
    plt.plot(y, x, label=l)
plt.legend()
plt.show()
