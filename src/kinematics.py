import numpy as np
import math as m
import random


"""
the first 3 coordinates tell which joints are connected in a linear fashion.
the 4 th coordinate tells the allowed angle movement is +ve or -ve in Elbow.
5th and 6th tell in what angles it must lie if a rotation is done.
All rotations in Z axis only
"""
CONNECTED_JOINTS_HANDS = np.array((
  [17, 18, 19, 1, 2, 115], #left hand,
  [25, 26, 27, 1, 2, 115]  # right hand

))

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return m.sqrt(dotproduct(v, v))

def angle_between(v1, v2):
  angle = m.acos(dotproduct(v1, v2) / (length(v1) * length(v2))) * 180 / m.pi
  #print(angle)
  return angle

def augment_kinematics(train_set):
  print("Kinematics is set to True...")
  augmented_kin = dict()
  for (sub, act, string), data in train_set.items():
    print("For :", sub, act, string)
    """
    # elbows
    data_new_elbows = apply_kinematics_elbow(data)
    print("Kinematicating elbows..", len(data_new_elbows))
    sname_e = string[:-3] + ".KIE.h5"
    augmented_kin.update({(sub, act, sname_e): data_new_elbows})
    """
    #total hands
    data_new_hands = apply_kinematics_hands(data)
    print("Kinematicating hands..", len(data_new_hands))
    sname_h = string[:-3] + ".KIH.h5"
    augmented_kin.update({(sub, act, sname_h): data_new_hands})

  return augmented_kin

#this method only moves the elbow
#rotate only the wrist around the elbows
#http://doc.aldebaran.com/2-1/family/robots/joints_robot.html
def apply_kinematics_elbow(frames):
  angles =np.array(([0]))
  data_new = []
  # radomly get how much degree you want to rotate around z
  for frame in frames:
    sk = np.array(frame)
    sk = np.reshape(sk, (-1, 3))
    # bring it to local frame
    for arm in CONNECTED_JOINTS_HANDS:
      root = sk[arm[0]]
      sk = sk - root
      deg = random.randint(0, 15) * arm[3] #determine if angle of rotation is postive or negative too
      #print("degree :", deg)
      angle = deg * m.pi / 180
      # arm will contain the joints
      # find the 2 vectors in the plane
      # both vectors start from the the 0th and
      # we rotate both of them
      v1 = sk[arm[1]] - sk[arm[0]]  # upper hand
      v2 = sk[arm[2]] - sk[arm[1]]  # elbow
      org_ang = angle_between(v1,v2)
      #print("old angle ", angle_between(v2,v1)* arm[3])
      #axis of rotation is the cross product of the vectors formed
      axis = np.cross(v1, v2)
      Rz = get_rot_matrix_around_vector(axis, angle)
      v2 = np.dot(Rz, v2)           # rotating the wrist around the elbow only
      #print("new angle ", angle_between(v2, v1)* arm[3])
      #check the angle in between the v1 and v2 and it
      #should be less than allowed
      angle_bet = angle_between(v1, v2) * arm[3] #angle between always return +ve angle and hence we multiply by the respective sign
      if not(arm[4] <= angle_bet and angle_bet <= arm[5] ):
        #reject these pair of joints and proceed
        #print("more than allowed rotation, not possible. Rejecting..", angle_bet)
        #print("original angle: ", org_ang)
        #print("degree :", deg)
        #np.append(angles, [org_ang])
        continue

      # now add the coordinate back
      v2 = v2 + sk[arm[1]]
      sk[arm[2]] = v2
      # add back to bring it to global
      sk = sk + root
    #import viz
    #viz.plot(frame, connect=True, c=viz.connect_points32())
    #viz.plot(sk.reshape((1, -1)), connect=True, c=viz.connect_points32())
    #viz.plotNoisySkeleton(frame, sk.reshape((1, -1)), [17,18])
    data_new.append(sk.reshape((1, -1)))
  #print(np.amax(angles))
  return data_new

def apply_kinematics_hands(frames):
  data_new = []
  #radomly get how much degree you want to rotate around z
  for frame in frames:
    sk = np.array(frame)
    sk = np.reshape(sk, (-1,3))
    #bring it to local frame
    root = sk[0]
    sk = sk - root
    for arm in CONNECTED_JOINTS_HANDS:
      deg = random.randint(-15, 15) * arm[3] * -1
      #print("degree :", deg)
      angle = deg * m.pi / 180
      Rz = np.array((
        [m.cos(angle), -m.sin(angle), 0],
        [m.sin(angle), m.cos(angle), 0],
        [0, 0, 1],
      ))
      #arm will contain the joints
      #find the 2 vectors in the plane
      #both vectors start from the the 0th and
      #we rotate both of them
      v1 = sk[arm[1]] - sk[arm[0]] #upper hand to wrist
      v2 = sk[arm[2]] - sk[arm[0]] #elbow to wrist
      v1 = np.dot(Rz, v1)
      v2 = np.dot(Rz, v2)
      #now add the coordinate back
      v1 = v1 + sk[arm[0]]
      v2 = v2 + sk[arm[0]]

      #new coordinates are sk[arm[0]], v1, v2
      sk[arm[1]] = v1
      sk[arm[2]] = v2
    #add back to bring it to global
    sk = sk + root
    #import viz
    #viz.plot(frame, connect=True, c=viz.connect_points32())
    #viz.plot(sk.reshape((1, -1)), connect=True, c=viz.connect_points32())
    data_new.append(sk.reshape((1,-1)))
    #import viz
    #viz.plotNoisySkeleton(frame, sk.reshape((1, -1)), [17, 18])
  return data_new

#vector around which you are rotating
#refer to https://steve.hollasch.net/cgindex/math/rotvec.html
def get_rot_matrix_around_vector(v, angle):
  #construct S matrix
  v = v.T/np.linalg.norm(v)
  x, y, z = v
  v = np.reshape(v, (3,-1))
  S = np.array((
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0]
  ))
  I = np.identity(3)
  R = v*v.T + m.cos(angle) * (I - v*v.T) + m.sin(angle) * S
  return R
