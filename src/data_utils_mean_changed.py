
"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import os
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import cameras
#import viz
import h5py
import glob
import copy
import math as m
import random
import kinematics
import data_utils_org as duo

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1,5,6,7,8]
TEST_SUBJECTS  = [9,11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

def load_data( bpath, subjects, actions, dim=3, rcams=None ):
  """
  Loads 2d ground truth from disk, and puts it in an easy-to-acess dictionary

  Args
    bpath: String. Path where to load the data from
    subjects: List of integers. Subjects whose data will be loaded
    actions: List of strings. The actions to load
    dim: Integer={2,3}. Load 2 or 3-dimensional data
  Returns:
    data: Dictionary with keys k=(subject, action, seqname)
      values v=(nx(32*2) matrix of 2d ground truth)
      There will be 2 entries per subject/action if loading 3d data
      There will be 8 entries per subject/action if loading 2d data

  Sid:
  subtract the average camera position to center the data around the space
  """

  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  avgCamPos = np.array(([0],[0],[0]))
  #Sid: added this line
  if rcams != None:
    avgCamPos = cameras.find_avg_cam_pos(rcams)

  data = {}

  for subj in subjects:
    for action in actions:

      print('Reading subject {0}, action {1}'.format(subj, action))

      dpath = os.path.join( bpath, 'S{0}'.format(subj), 'MyPoses/{0}D_positions'.format(dim), '{0}*.h5'.format(action) )
      #print( dpath )

      fnames = glob.glob( dpath )

      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )

        # This rule makes sure SittingDown is not loaded when Sitting is requested
        if action == "Sitting" and seqname.startswith( "SittingDown" ):
          continue

        # This rule makes sure that WalkDog and WalkTogeter are not loaded when
        # Walking is requested.
        if seqname.startswith( action ):
          #print( fname )
          loaded_seqs = loaded_seqs + 1

          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f['{0}D_positions'.format(dim)][:]

          poses = poses.T

          #Sid: added this block to move world data to avg cam position
          newPoses = []
          for row in poses:
            k = np.reshape(row,(-1,3))
            k = k - avgCamPos.T
            k = np.reshape(k,(1,-1))
            newPoses.append(k)

          data[ (subj, action, seqname) ] = newPoses #poses

      if dim == 2:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead".format( loaded_seqs )
      else:
        assert loaded_seqs == 2, "Expecting 2 sequences, found {0} instead".format( loaded_seqs )

  return data


def load_stacked_hourglass(data_dir, subjects, actions):
  """
  Load 2d detections from disk, and put it in an easy-to-acess dictionary.

  Args
    data_dir: string. Directory where to load the data from,
    subjects: list of integers. Subjects whose data will be loaded.
    actions: list of strings. The actions to load.
  Returns
    data: dictionary with keys k=(subject, action, seqname)
          values v=(nx(32*2) matrix of 2d stacked hourglass detections)
          There will be 2 entries per subject/action if loading 3d data
          There will be 8 entries per subject/action if loading 2d data
  """
  # Permutation that goes from SH detections to H36M ordering.
  SH_TO_GT_PERM = np.array([SH_NAMES.index( h ) for h in H36M_NAMES if h != '' and h in SH_NAMES])
  assert np.all( SH_TO_GT_PERM == np.array([6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10]) )

  data = {}

  for subj in subjects:
    for action in actions:

      print('Reading subject {0}, action {1}'.format(subj, action))

      dpath = os.path.join( data_dir, 'S{0}'.format(subj), 'StackedHourglass/{0}*.h5'.format(action) )
      print( dpath )

      fnames = glob.glob( dpath )

      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )
        seqname = seqname.replace('_',' ')

        # This rule makes sure SittingDown is not loaded when Sitting is requested
        if action == "Sitting" and seqname.startswith( "SittingDown" ):
          continue

        # This rule makes sure that WalkDog and WalkTogeter are not loaded when
        # Walking is requested.
        if seqname.startswith( action ):
          print( 'file :',fname )
          loaded_seqs = loaded_seqs + 1

          # Load the poses from the .h5 file
          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f['poses'][:]

            # Permute the loaded data to make it compatible with H36M
            poses = poses[:,SH_TO_GT_PERM,:]

            # Reshape into n x (32*2) matrix
            poses = np.reshape(poses,[poses.shape[0], -1])
            poses_final = np.zeros([poses.shape[0], len(H36M_NAMES)*2])

            dim_to_use_x    = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
            dim_to_use_y    = dim_to_use_x+1

            dim_to_use = np.zeros(len(SH_NAMES)*2,dtype=np.int32)
            dim_to_use[0::2] = dim_to_use_x
            dim_to_use[1::2] = dim_to_use_y
            poses_final[:,dim_to_use] = poses
            seqname = seqname+'-sh'
            data[ (subj, action, seqname) ] = poses_final

      # Make sure we loaded 8 sequences
      if (subj == 11 and action == 'Directions'): # <-- this video is damaged
        assert loaded_seqs == 7, "Expecting 7 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )
      else:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )

  return data


def normalization_stats(complete_data, dim, predict_14=False ):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    complete_data: nxd np array with poses
    dim. integer={2,3} dimensionality of the data
    predict_14. boolean. Whether to use only 14 joints
  Returns
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions not used in the model
    dimensions_to_use: list of dimensions used in the model
  """
  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data_mean = np.mean(complete_data, axis=0)
  data_std  =  np.std(complete_data, axis=0)

  # Encodes which 17 (or 14) 2d-3d pairs we are predicting
  dimensions_to_ignore = []
  if dim == 2:
    dimensions_to_use    = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
    dimensions_to_use    = np.sort( np.hstack( (dimensions_to_use*2, dimensions_to_use*2+1)))
    dimensions_to_ignore = np.delete( np.arange(len(H36M_NAMES)*2), dimensions_to_use )
  else: # dim == 3
    dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
    dimensions_to_use = np.delete( dimensions_to_use, [0,7,9] if predict_14 else 0 )

    dimensions_to_use = np.sort( np.hstack( (dimensions_to_use*3,
                                             dimensions_to_use*3+1,
                                             dimensions_to_use*3+2)))
    dimensions_to_ignore = np.delete( np.arange(len(H36M_NAMES)*3), dimensions_to_use ) #Sid, since this is a vector, all the x,y,z is multiplied by 3 and added 1

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def transform_world_to_camera(poses_set, cams, ncams=4 ):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted( poses_set.keys() ):

      subj, action, seqname = t3dk
      t3d_world = poses_set[ t3dk ]

      for c in range( ncams ):
        R, T, f, c, k, p, name = cams[ (subj, c+1) ]
        camera_coord = cameras.world_to_camera_frame( np.reshape(t3d_world, [-1, 3]), R, T)
        camera_coord = np.reshape( camera_coord, [-1, len(H36M_NAMES)*3] )

        sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
        t3d_camera[ (subj, action, sname) ] = camera_coord

    return t3d_camera


def normalize_data(data, data_mean, data_std, dim_to_use ):
  """
  Normalizes a dictionary of poses

  Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions to keep in the data
  Returns
    data_out: dictionary with same keys as data, but values have been normalized
  """
  data_out = {}

  for key in data.keys():
    data[ key ] = data[ key ][ :, dim_to_use ] #Sid, using here only the columms that are required
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out[ key ] = np.divide( (data[key] - mu), stddev )

  return data_out


def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_ignore):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing

  Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
  Returns
    orig_data: the input normalized_data, but unnormalized
  """
  T = normalized_data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality

  orig_data = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = np.array([dim for dim in range(D)
                                if dim not in dimensions_to_ignore])

  orig_data[:, dimensions_to_use] = normalized_data

  # Multiply times stdev and add the mean
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  orig_data = np.multiply(orig_data, stdMat) + meanMat
  return orig_data


def define_actions( action ):
  """
  Given an action string, returns a list of corresponding actions.

  Args
    action: String. either "all" or one of the h36m actions
  Returns
    actions: List of strings. Actions to use.
  Raises
    ValueError: if the action is not a valid action in Human 3.6M
  """
  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all":
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]

def project_to_cameras( poses_set, cams, ncams=4 ):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    ncams: number of cameras per subject
  Returns
    t2d: dictionary with 2d poses
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    for cam in range( ncams ):
      R, T, f, c, k, p, name = cams[ (subj, cam+1) ]
      pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3d, [-1, 3]), R, T, f, c, k, p )

      pts2d = np.reshape( pts2d, [-1, len(H36M_NAMES)*2] )
      sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
      t2d[ (subj, a, sname) ] = pts2d

  return t2d

def project_to_cameras2( poses_set, cams, ncams=4, no_of_joints=len(H36M_NAMES)):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    ncams: number of cameras per subject
  Returns
    t2d: dictionary with 2d poses
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    for cam in range( ncams ):
      R, T, f, c, k, p, name = cams[ (subj, cam+1) ]
      pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3d, [-1, 3]), R, T, f, c, k, p )

      pts2d = np.reshape( pts2d, [-1, no_of_joints * 2] )
      if name not in seqname:
        sname = seqname[:-3] + "." + name + ".h5"  # e.g.: Waiting 1.58860488.h5
      else:
        sname = seqname #Sid, when this method is called for noisy data, the keys are already created and so not needed to change.
      t2d[ (subj, a, sname) ] = pts2d

  return t2d


def read_2d_predictions( actions, data_dir ):
  """
  Loads 2d data from precomputed Stacked Hourglass detections

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
  Returns
    train_set: dictionary with loaded 2d stacked hourglass detections for training
    test_set: dictionary with loaded 2d stacked hourglass detections for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """

  train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions)
  test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS,  actions)

  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std,  dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def create_2d_data( actions, data_dir, rcams, augmented3d=None ):
  """
  Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters. Also normalizes the 2d poses

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    rcams: dictionary with camera parameters
  Returns
    train_set: dictionary with projected 2d poses for training
    test_set: dictionary with projected 2d poses for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """

  # Load 3d data
  train_set = duo.load_data( data_dir, TRAIN_SUBJECTS, actions, dim=3 )
  test_set  = duo.load_data( data_dir, TEST_SUBJECTS,  actions, dim=3 )

  train_set = duo.project_to_cameras( train_set, rcams) #for 5 cams put here
  test_set  = project_to_cameras( test_set, rcams )

  if augmented3d != None:
    augmented2d = project_to_cameras(augmented3d,rcams)
    #train_set = {**train_set, **augmented2d}
    train_set.update(augmented2d)

  train_set_2d_for_noisy = train_set.copy()

  # Compute normalization statistics.
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )
  complete_test = copy.deepcopy(np.vstack( test_set.values() ))
  data_mean_test, data_std_test, dim_to_ignore, dim_to_use = normalization_stats(complete_test, dim=2)


  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean_test, data_std_test, dim_to_use )

  print("For normal, mean :")
  print(data_mean)
  print("For normal, std :")
  print(data_std)
  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_set_2d_for_noisy, data_mean_test, data_std_test



def read_3d_data( actions, data_dir, camera_frame, rcams, predict_14=False, augment_all=False, augment_rot=False, augment_flip=False, augment_trans=False, noisy=False, add_kinematics=False ):
  """
  Loads 3d poses, zero-centres and normalizes them

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    camera_frame: boolean. Whether to convert the data to camera coordinates
    rcams: dictionary with camera parameters
    predict_14: boolean. Whether to predict only 14 joints
  Returns
    train_set: dictionary with loaded 3d poses for training
    test_set: dictionary with loaded 3d poses for testing
    data_mean: vector with the mean of the 3d training data
    data_std: vector with the standard deviation of the 3d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
    train_root_positions: dictionary with the 3d positions of the root in train
    test_root_positions: dictionary with the 3d positions of the root in test
  """
  # Load 3d data
  train_set = duo.load_data( data_dir, TRAIN_SUBJECTS, actions, dim=3 )
  test_set  = duo.load_data( data_dir, TEST_SUBJECTS,  actions, dim=3 )

  augmented3d = {}
  train_set_for_noisy = {}

  if noisy > 0:
    #keep a data copy for further
    train_set_for_noisy = train_set.copy()

  if augment_all == True or augment_rot == True or augment_flip == True or augment_trans == True:
    augmented3d = augment3Ddata(train_set, rcams, augment_all=augment_all, augment_rot=augment_rot, augment_flip=augment_flip, augment_trans=augment_trans)
    #train_set = {**train_set, **augmented3d}
    train_set.update(augmented3d)

  if augment_all == True or add_kinematics == True:
    aug_kinematics = kinematics.augment_kinematics(train_set)
    augmented3d.update(aug_kinematics)

  train_set.update(augmented3d)


  if camera_frame:
    train_set = transform_world_to_camera( train_set, rcams )
    test_set  = transform_world_to_camera( test_set, rcams )

  # Apply 3d post-processing (centering around root)
  train_set, train_root_positions = postprocess_3d( train_set )
  test_set,  test_root_positions  = postprocess_3d( test_set )

  # Compute normalization statistics
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=3, predict_14=predict_14 )
  complete_test = copy.deepcopy(np.vstack(test_set.values()))
  data_mean_test, data_std_test, dim_to_ignore, dim_to_use = normalization_stats(complete_test, dim=3, predict_14=predict_14)

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean_test, data_std_test, dim_to_use )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions, augmented3d, train_set_for_noisy, data_mean_test, data_std_test


def postprocess_3d( poses_set ):
  """
  Center 3d points around root

  Args
    poses_set: dictionary with 3d data
  Returns
    poses_set: dictionary with 3d data centred around root (center hip) joint
    root_positions: dictionary with the original 3d position of each pose
  """
  root_positions = {}
  for k in poses_set.keys():
    # Keep track of the global position
    root_positions[k] = copy.deepcopy(poses_set[k][:,:3])

    # Remove the root from the 3d position
    poses = poses_set[k]
    poses = poses - np.tile( poses[:,:3], [1, len(H36M_NAMES)] ) #Sid, subtracting the root node from all joints
    poses_set[k] = poses

  return poses_set, root_positions


def augment3Ddata(train_set_3d, rcams, augment_all=False, augment_rot=False, augment_flip=False, augment_trans=False,flipAxis = 'XY'):
 """
  passed a 3d set it should augment
  the data with arbitary 3d rotations
  of the joints
 """
 print("augmenting 3D data, all augmentation(R+T+F) =",augment_all,
       ", with rotation = ", augment_rot, ",Flip = ", augment_flip, ", Translated =",augment_trans )
 tst3d = train_set_3d
 augmented = dict()
 rot_count = 0
 flip_count = 0
 sub_avg_cam_dict = {}
 for (sub,act,string),data in tst3d.items():
  #print(act)
  dataFlipped = np.empty([1,96])
  if augment_all == True or augment_rot == True:
    rand = random.randint(-90, 180)
    print("Rotating subject ", sub, "Action ",act, "by : ", rand)
    dataRotated = rotate3DTrajectory3(data, rand)
    rot_count = rot_count + data.shape[0]
    #viz.plotTrail(data, dataRotated)
    sname = string[:-3] + ".ROT.h5"
    augmented.update({(sub, act, sname): dataRotated})

  if augment_all == True or augment_flip == True:
    # for flipping around avg camera
    if sub in sub_avg_cam_dict:
      avg = sub_avg_cam_dict[sub]
    else:
      avg = cameras.find_avg_cam_pos_per_subj(rcams, sub)
      sub_avg_cam_dict[sub] = avg
      print("avg :", avg)

    for frame in data:
      #dataFlipped.append(flip3Ddata(frame, flipAxis, type='local'))
      flip = flip3Ddata2(avg, frame)
      #dataFlipped = np.append(dataFlipped, flip)h
      dataFlipped = np.vstack([dataFlipped, flip])
      #import viz
      #viz.plot(frame, connect=True, c=viz.connect_points32())
      #viz.plot(flip, connect=True, c=viz.connect_points32())
      flip_count = flip_count + 1
    sname = string[:-3] + ".FLIP."+flipAxis+".h5"
    augmented.update({(sub,act,sname):dataFlipped})
    #import viz
    #viz.plotTrail2(data, dataFlipped, 'Original', 'Flipped data')
 print("Rotated :", rot_count, "Flip count :", flip_count)
 print("done with rotation/flip/translation" )
 return augmented


trail = []
trailRot = []
c = 0
trailPlot = False
_3dplot = True
points = 15
indx = 11




def rotate3DTrajectory3(arrayOfPoints, deg):
  rad = deg * m.pi / 180
  aofp = np.array(arrayOfPoints, dtype=float)
  aofp = np.reshape(aofp,(-1,3)).T

  # should be every 32nd node
  # find the root node, evry 32,33,34th value
  rootNodes = aofp[:,0::32]
  avg = np.reshape(np.average(rootNodes, axis=1), (3,-1))
  avg = np.tile(avg, [aofp.shape[1]])
  naofp = aofp - avg
  #prepare for matrix multiplication
  #naofp = np.reshape(naofp , (-1,3)).T
  #now rotate
  R = np.array((
    [m.cos(rad), -m.sin(rad), 0],
    [m.sin(rad), m.cos(rad), 0],
    [0, 0, 1]
  ))
  naofp = np.matmul(R, naofp) + avg #now this has 3 rows
  #need to transpose and flatten this
  #naofp = naofp.T
  naofp = np.reshape(naofp.T, (1,-1))
  #print("a batch rotated...")
  #import viz
  #viz.plotTrail2(arrayOfPoints, naofp, 'Original Trajectory','Trajectory after rotating by '+str(deg)+" degrees")
  return naofp

def flip3Ddata2(avg, arrayOfPoints):

  #find the avg position of the cameras
  avg = np.reshape(avg, (3,-1))
  aofp = np.array(arrayOfPoints, dtype=float)
  aofp = np.reshape(aofp, (-1, 3)).T

  #subtract the avg pos them from the frame
  avg = np.tile(avg, [aofp.shape[1]])
  naofp = aofp - avg
  #flip the data around some plane for ex XZ
  v = np.array(([1],[-1],[1]))
  naofp = naofp * v
  #add the avg pos back
  naofp = naofp + avg
  #return flipped data
  return np.reshape(naofp.T,(1,-1))



