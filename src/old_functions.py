def augment3Ddata2(train_set_3d, augment_all=False, augment_rot=False, augment_flip=False, augment_trans=False,flipAxis = 'XY', angle=180):
 """
  passed a 3d set it should augment
  the data with arbitary 3d rotations
  of the joints
 """
 print("augmenting 3D data, all augmentation(R+T+F) =",augment_all,
       ", with rotation = ", augment_rot, ",Flip = ", augment_flip, ", Translated =",augment_trans )
 tst3d = train_set_3d
 augmented = dict()
 for (sub,act,string),data in tst3d.items():
  #print(act)
  dataRotated = []
  dataFlipped = []
  if augment_all == True or augment_rot == True:

    for frame in data:
      dataRotated.append(rotate3dData(frame, angle, type='local'))
    sname = string[:-3] + ".ROT.h5"
    #string = string + '.Rot'
    augmented.update({(sub,act,sname):dataRotated})

  if augment_all == True or augment_flip == True:
    for frame in data:
      dataFlipped.append(flip3Ddata(frame, flipAxis, type='local'))
    sname = string[:-3] + ".FLIP."+flipAxis+".h5"
    augmented.update({(sub,act,sname):dataFlipped})

 return augmented


def rotate3dData(arrayOf3Djoints, angle, type='local'):
  angle = angle * m.pi / 180
  newArrayOf3Djoints = list()
  if _3dplot == True:
    viz.plot(arrayOf3Djoints)
  arrayOf3Djoints = np.reshape(arrayOf3Djoints, (-1, 3))

  #if local rotation average and do rotation alon center of mass
  avg = sum(arrayOf3Djoints)/len(arrayOf3Djoints)

  #rotating around Z axis by angle angle
  indx = 0
  for row in arrayOf3Djoints:
    x,y,z = row[0],row[1],row[2]
    xc, yc, zc = x, y, z

    if type == 'local':
      #subtract to bring to local co-ordinate system
      x, y, z = x - avg[0], y - avg[1], z - avg[2]


    x_ = x*m.cos(angle) - y*m.sin(angle)
    y_ = x*m.sin(angle) + y*m.cos(angle)
    z_ = z

    if type == 'local':
      #add back to bring it back to world
      x_,y_,z_ = x_ + avg[0], y_ + avg[1], z_ + avg[2]

    newArrayOf3Djoints.append(x_)
    newArrayOf3Djoints.append(y_)
    newArrayOf3Djoints.append(z_)

    #ploting trail code
    if trailPlot ==  True:
      global c,trail, trailRot
      if c < points:
        if indx == 0:
          trail.append((xc,yc))
          trailRot.append((x_,y_))
          c = c+1
          if c == points:
            viz.plotTrail(trail, trailRot)
            c = 0
            trail = []
            trailRot = []

    indx = indx + 1
    #plotting trail code ends.

  if _3dplot == True:
    viz.plot(newArrayOf3Djoints)

  return newArrayOf3Djoints


def rotate3DTrajectory(aofp, angle):
  angle = angle * m.pi / 180
  aofp = np.reshape(aofp, (3,-1))
  #average all points
  avg = np.average(aofp, axis=1)
  #substract that from trajectory
  nwofp = aofp - np.tile(avg, (3, aofp.shape[1]))
  #now, rotate
  R = np.array((
    [m.cos(angle), -m.sin(angle), 0],
    [m.sin(angle), m.cos(angle), 0],
    [0, 0, 1]
  ))
  nwofp = np.matmul(R, nwofp)
  #add back avg
  nwofp = nwofp + np.tile(avg, (3, aofp.shape[1]))
  return nwofp

"""
Array of points (trajectory) is given as input
and it should subtract the root node i.e., the 0th element from
tha subset 
"""
def rotate3DTrajectory2(aofp, angle):
  angle = angle * m.pi / 180
  naofp = []
  rootNodes = []
  #get all root nodes
  for frame in aofp:
    rn = frame[:,3]
    rootNodes = vstack[(rootNodes,rn)]
  #average it
  avg = np.average(rootNodes, axis=1)
  for frame in aofp:
    frame = frame - np.tile(avg, len(frame)/len(avg))
    naofp.append(frame)
  aofp = np.array((aofp))
  aofp = np.reshape(aofp, (1,-1))
  # should be every 32nd node
  # find the root node, evry 32,33,34th value
  rootNodes = np.array[[
    aofp[0::32],
    aofp[1::32],
    aofp[2::32]
  ]]
  avg = np.average(rootNodes,axis=1)
  aofp = np.reshape(aofp, (3, -1))
  #scale average to same size as data so that u can subtract
  # substract that from trajectory
  nwofp = aofp - np.tile(avg, (3, aofp.shape[1]))
  # now, rotate
  R = np.array((
    [m.cos(angle), -m.sin(angle), 0],
    [m.sin(angle), m.cos(angle), 0],
    [0, 0, 1]
  ))
  nwofp = np.matmul(R, nwofp)
  # add back avg
  nwofp = nwofp + np.tile(avg, (3, aofp.shape[1]))
  return nwofp

def flip3Ddata(arrayOf3Djoints, plane='XZ', type='local'):

  if _3dplot == True:
    viz.plot(arrayOf3Djoints)

  indx = 0
  newArrayOf3Djoints = list()
  arrayOf3Djoints = np.reshape(arrayOf3Djoints, (-1, 3))

  # if local rotation average and do rotation alon center of mass
  avg = sum(arrayOf3Djoints) / len(arrayOf3Djoints)

  for row in arrayOf3Djoints:
    x,y,z = row[0],row[1],row[2]
    xc,yc,zc = x,y,z

    #global to local
    if type == 'local':
      # subtract to bring to local co-ordinate system
      x, y, z = x - avg[0], y - avg[1], z - avg[2]

    if plane == 'XZ':
      x_ = x
      y_ = -y
      z_ = z
    else:
      x_ = -x
      y_ = y
      z_ = z

    #local to global
    if type == 'local':
      #add back to bring it back to world
      x_,y_,z_ = x_ + avg[0], y_ + avg[1], z_ + avg[2]

    newArrayOf3Djoints.append(x_)
    newArrayOf3Djoints.append(y_)
    newArrayOf3Djoints.append(z_)

    # ploting trail code
    if trailPlot == True:
      global c, trail, trailRot
      if c < points:
        if indx == 11:
          trail.append((xc, yc))
          trailRot.append((x_, y_))
          c = c + 1
          if c == points:
            plotTrail(trail, trailRot)
            c = 0
            trail = []
            trailRot = []

    indx = indx + 1
    # plotting trail code ends.

  if _3dplot == True:
    viz.plot(newArrayOf3Djoints)
  return newArrayOf3Djoints



#has now 16 joints

def addNoiseTo3D(train_set_3d):
  noise_deg = np.array((
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
  ))
  mu = 0
  #viz.show3Dpose(train_set_3d[0], ax)
  for (sub, act, string), frames in train_set_3d.items():
    dataCorrupted = []
    for frame in frames:
      #take and do something with the frame.
      #randomly pick 4 joints and corrupt them

      #prepare for multiplication
      data = np.reshape(frame, (-1,3))
      k = random.randint(0,4) #random.sample([0,0,0,1], k=1)
      j_sampled = random.sample(range(0, noise_deg.shape[0]), k=k) #randomly take 0 - 4 joints
      vec = np.zeros(noise_deg.shape[0])
      #this will change the vec to same value as noise_deg in those positions
      vec = np.tile(vec, [3, 1]).T
      for j in j_sampled:
        vec[j] = np.random.normal(mu, noise_deg[j], 3)
      data = data + vec
      #data = list(np.reshape(data, (1,-1)))
      data = np.reshape(data, (-1))
      dataCorrupted.append(data)
      #import viz
      #viz.plotNoisySkeleton(frame, data, j_sampled)
    train_set_3d.update({(sub, act, string): np.asarray(dataCorrupted)})

  return train_set_3d

def perform_noisy_actions(train_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, rcams):
  print("Noisy data is on. Adding noise.py to the data.")
  train_set_3d_copy = copy.deepcopy(train_set_3d)
  train_set_3d_copy = unNormalizeData3(train_set_3d_copy, data_mean_3d, data_std_3d, dim_to_ignore_3d)
  # add noise.py
  #train_set_3d_copy = addNoiseTo3D(train_set_3d_copy)
  # normalise back with new mean and std dev
  complete_train = copy.deepcopy(np.vstack(train_set_3d_copy.values()))
  data_mean, data_std, dimensions_to_ignore, dimensions_to_use = normalization_stats(complete_train, 3)
  train_set_3d_copy = normalize_data2(train_set_3d_copy, data_mean, data_std)
  # now create 2D projections for them
  train_set_2d, data_mean_2d, data_std_2d = create_2d_data_noisy(
    rcams, train_set_3d_copy)
  return train_set_2d, data_mean_2d, data_std_2d

def create_2d_data_noisy(rcams, noisy3dData):
  noisy2dData = project_to_cameras(noisy3dData, rcams, no_of_joints=16)
  # Compute normalization statistics.
  complete_train = copy.deepcopy( np.vstack( noisy2dData.values() ))
 #data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )
  data_mean = np.mean(complete_train, axis=0)
  data_std = np.std(complete_train, axis=0)
  dim_to_use = np.array((range(0,32)))

  # Divide every dimension independently
  train_set = normalize_data( noisy2dData, data_mean, data_std, dim_to_use )

  return train_set, data_mean, data_std

def unNormalizeData2(normalized_data, data_mean, data_std, dimensions_to_ignore):
  """
  difference from unNormalizeData
  1. Doesnt change the size of the array
  """
  data_mean = np.delete(data_mean, dimensions_to_ignore )
  data_std = np.delete(data_std, dimensions_to_ignore)
  T = normalized_data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality

  orig_data = normalized_data

  # Multiply times stdev and add the mean
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  orig_data = np.multiply(orig_data, stdMat) + meanMat
  return orig_data

def unNormalizeData3(data, data_mean, data_std, dimensions_to_ignore):
  data_mean = np.delete(data_mean, dimensions_to_ignore)
  data_std = np.delete(data_std, dimensions_to_ignore)
  for key in data.keys():
   normalized_data = data[key]
   T = normalized_data.shape[0]  # Batch size
   D = data_mean.shape[0]  # Dimensionality

   orig_data = normalized_data

   # Multiply times stdev and add the mean
   stdMat = data_std.reshape((1, D))
   stdMat = np.repeat(stdMat, T, axis=0)
   meanMat = data_mean.reshape((1, D))
   meanMat = np.repeat(meanMat, T, axis=0)
   orig_data = np.multiply(orig_data, stdMat) + meanMat

   data.update({key:orig_data})

  return data

def normalize_data2(data, data_mean, data_std):
  """
  Normalizes a dictionary of poses

  Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
  Returns
    data_out: dictionary with same keys as data, but values have been normalized

  Difference from normalise_data(...) doesnt have dim to ignore, normalises whatever is given
  """
  data_out = {}

  for key in data.keys():
    data[ key ] = data[ key ]
    mu = data_mean
    stddev = data_std
    data_out[ key ] = np.divide( (data[key] - mu), stddev )

  return data_out
