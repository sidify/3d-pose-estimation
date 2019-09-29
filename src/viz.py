
"""Functions to visualize human poses"""

#import matplotlib.pyplot as plt
import data_utils
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False): # blue, orange
  """
  Visualize a 3d skeleton

  Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )

  I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
  J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
  LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)
    #ax.text(x,y,z,i)

  RADIUS = 750 # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

  # Get rid of the ticks and tick labels
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])

  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
  ax.set_zticklabels([])
  ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)

def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
  """
  Visualize a 2d skeleton

  Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(data_utils.H36M_NAMES)*2, "channels should have 64 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )

  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
    ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

  # Get rid of the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Get rid of tick labels
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

  RADIUS = 350 # space around the subject
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")

  ax.set_aspect('equal')



def connect_points2():
    """Sid.
    This functions only visualises to connect points
    for 16 joints only
    """
    connector = np.array((
      [6, 1, -1, -1],  # 0
      [2, -1, -1, -1],  # 1
      [-1, -1, -1, -1],  # 2
      [6, 4, -1, -1],  # 3
      [3, 5, -1, -1],  # 4
      [-1, -1, -1, -1],  # 5
      [0, 3, 7, -1],  # 6
      [8, 10, 13, 6],  # 7
      [9, 7, -1, -1],  # 8
      [-1, -1, -1, -1],  # 9
      [7, 11, -1, -1],  # 10
      [12, -1, -1, -1],  # 11
      [-1, -1, -1, -1],  # 12
      [7, 14, -1, -1],  # 13
      [13, 15, -1, -1],  # 14
      [-1, -1, -1, -1]  # 15
    ))

    return connector

def connect_points32():
    """Sid.
    This functions only visualises to connect points
    for 16 joints only
    """
    connector = np.array((
      [6, 1, 12, -1],  # 0
      [2, 0, -1, -1],  # 1
      [1, 3, -1, -1],  # 2
      [2, -1, -1, -1],  # 3
      [-1, -1, -1, -1],  # 4
      [-1, -1, -1, -1],  # 5
      [0, 7, -1, -1],  # 6
      [6, 8, -1, -1],  # 7
      [7, -1, -1, -1],  # 8
      [-1, -1, -1, -1],  # 9
      [-1, -1, -1, -1],  # 10
      [-1, -1, -1, -1],  # 11
      [0, 13, -1, -1],  # 12
      [12, 17, 25, 14],  # 13
      [13, 15, -1, -1],  # 14
      [-1, -1, -1, -1],  # 15
      [-1, -1, -1, -1],  # 16
      [18, 13, -1, -1],  # 17
      [19, -1, -1, -1],  # 18
      [-1, -1, -1, -1],  # 19
      [-1, -1, -1, -1],  # 20
      [-1, -1, -1, -1],  # 21
      [-1, -1, -1, -1],  # 22
      [-1, -1, -1, -1],  # 23
      [-1, -1, -1, -1],  # 24
      [13, 26, -1, -1],  # 25
      [27, 25, -1, -1],  # 26
      [-1, -1, -1, -1],  # 27
      [-1, -1, -1, -1],  # 28
      [-1, -1, -1, -1],  # 29
      [-1, -1, -1, -1],  # 30
      [-1, -1, -1, -1],  # 31

    ))

    return connector
def plotNoisySkeleton(skeleton1, skeleton2, changed):
    fig = plt.figure()
    connect = True
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-750, 750)
    ax.set_ylim3d(-750, 750)
    ax.set_zlim3d(0, 2000)
    col = ['g', 'r']
    count = 0
    for skeleton in [skeleton1, skeleton2]:
        data = np.reshape(skeleton, (-1, 3))
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        #ax.scatter(x, y, z, c='b')
        for i in range(data.shape[0]):
            #ax.text(x[i], y[i], z[i], i)
            if connect == True:
                # find the child of the current point
                c = connect_points32()
                for child in c[i, :]:
                    if child == -1:
                        continue
                    # otherwise fetch that point from data
                    #print(i, child)
                    x_c, y_c, z_c = data[child, :]
                    ax.plot([x[i], x_c], [y[i], y_c], [z[i], z_c], c = col[count])
        count = 1
    plt.xlabel('X')
    plt.ylabel('Y')
    changed = ','.join(str(x) for x in changed)
    plt.title('Joints changed :'+ changed)
    plt.show()

def plotFlip(models, connect=False, c=connect_points2()):
    """Just plots  3dpoints and what connector to use"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1000, 1000)
    ax.set_ylim3d(-1000, 1000)
    ax.set_zlim3d(0, 2000)
    for model in models:
        data = np.reshape(model, (-1, 3))
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        #ax.scatter(x, y, z, c='r')
        for i in range(data.shape[0]):
          #ax.text(x[i], y[i], z[i], i)
          if connect == True:
            # find the child of the current point
            #c = connect_points2()
            for child in c[i, :]:
              if child == -1:
                continue
              # otherwise fetch that point from data
              #print(i, child)
              x_c, y_c, z_c = data[child, :]
              ax.plot([x[i], x_c], [y[i], y_c], [z[i], z_c])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot(arrayOf3Djoints, connect=False, c=connect_points2()):
    """Just plots  3dpoints and what connector to use"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1000, 1000)
    ax.set_ylim3d(-1000, 1000)
    ax.set_zlim3d(0, 2000)
    data = np.reshape(arrayOf3Djoints, (-1, 3))
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    #ax.scatter(x, y, z, c='r')
    for i in range(data.shape[0]):
      #ax.text(x[i], y[i], z[i], i)
      if connect == True:
        # find the child of the current point
        #c = connect_points2()
        for child in c[i, :]:
          if child == -1:
            continue
          # otherwise fetch that point from data
          #print(i, child)
          x_c, y_c, z_c = data[child, :]
          ax.plot([x[i], x_c], [y[i], y_c], [z[i], z_c])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plotTrail(trail, trailRot):
      """"show trailing plots of points in different frames after rotation"""
      trail = np.reshape(trail[:50], (-1, 2))
      trailRot = np.reshape(trailRot[:50], (-1, 2))
      fig, ax = plt.subplots()

      ax.scatter(trail[:, 0], trail[:, 1], c='r')
      ax.plot(trail[:, 0], trail[:, 1])

      #  for i in range(trail.shape[0]):
      #    ax.text(trail[:,0][i], trail[:,1][i],i)

      ax.scatter(trailRot[:, 0], trailRot[:, 1], c='b')
      ax.plot(trailRot[:, 0], trailRot[:, 1])

      #  for i in range(trailRot.shape[0]):
      #    ax.text(trailRot[:,0][i], trailRot[:,1][i],i)

      plt.show()

def plotTrail2(trail, trailRot, str, str2):
    fig, ax = plt.subplots()
    aofp = np.array(trail, dtype=float)
    aofp = np.reshape(aofp, (-1, 3)).T
    aofp2 = np.array(trailRot, dtype=float)
    aofp2 = np.reshape(aofp2, (-1, 3)).T


    # should be every 32nd node
    # find the root node, evry 32,33,34th value
    rootNodes = aofp[:, 0::32]
    ax.plot(rootNodes[0], rootNodes[1], c='r')

    rootNodes2 = aofp2[:, 0::32]
    ax.plot(rootNodes2[0], rootNodes2[1], c='b')

    plt.gca().legend((str,str2))
    plt.show()
