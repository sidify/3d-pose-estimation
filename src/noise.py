import data_utils as du
import numpy as np
import random
import copy
import data_utils_org as duo


def addNoiseTo3D(train_set_3d, percent):
  print("Adding noise to ", percent, "% of the data!")
  noise_deg = np.array((
    [0, 2, 15, 15, 0, 0, 2, 15, 15, 0, 0, 0, 2, 0, 2, 5, 0, 8, 15, 15, 0, 0, 0, 0, 0, 8, 15, 15, 0, 0, 0, 0]
    # [15,5,3,2,0,0,5,3,15,0,0,0,5,0,2,5,0,4,3,2,0,0,0,0,0,4,3,2,0,0,0,0]
    # 0   1  2  3  4  5  6  7   8  9  0  1  2    3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8  9  0  1
    # [15, 5, 3, 2, 0, 0, 5, 3, 2, 0, 0, 0, 15, 15, 3, 3, 0, 4, 2, 1, 0, 0, 0, 0, 0, 4, 2, 1, 0, 0, 0, 0]
    #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ))
  mu = 0
  selection = np.zeros(100)
  selected = random.sample(range(0,100), k=percent)
  for x in selected:
    selection[x] = 1
  #viz.show3Dpose(train_set_3d[0], ax)
  framesCorrupted = 0
  totalFrames = 0
  for (sub, act, string), frames in train_set_3d.items():
    dataCorrupted = []
    for frame in frames:
      totalFrames = totalFrames + 1
      #first see if we want to corrupt this frame
      k = random.randint(0, 99)
      # prepare for multiplication
      data = np.reshape(frame, (-1, 3)).copy()
      if selection[k] == 1:
        #we are randomly deciding to corrupt this frame

        #take and do something with the frame.
        #randomly pick 4 joints and corrupt them
        for i in range(0, data.shape[0], noise_deg.shape[0]):
          k = random.randint(4,8) #random.sample([0,0,0,1], k=1)
          j_sampled = random.sample(range(0, noise_deg.shape[0]), k=k) #randomly take 0 - 4 joints
          vec = np.zeros(noise_deg.shape[0])
          #this will change the vec to same value as noise_deg in those positions
          vec = np.tile(vec, [3, 1]).T
          for j in j_sampled:
            vec[j] = np.random.normal(mu, noise_deg[j], 3)
          data[i:i+noise_deg.shape[0]] = data[i:i+noise_deg.shape[0]] + vec
          framesCorrupted = framesCorrupted + 1
        #data = list(np.reshape(data, (1,-1)))
      data = np.reshape(data, (1,-1))
      #if not np.array_equal(frame, data):
      #  print("Alert!")
      dataCorrupted.append(data)
      import viz
      #viz.plotFlip([frame,data], connect=True, c=viz.connect_points32())
      viz.plotNoisySkeleton(frame, data, j_sampled)
      #viz.plotFlipe([frame,data], )
    train_set_3d.update({(sub, act, string): np.asarray(dataCorrupted)})

  print("Total frames : ", totalFrames)
  print("Corrupted frames :", framesCorrupted)
  print("Percentage: ", framesCorrupted / totalFrames * 100, "%")
  return train_set_3d


#summation of create_3d_data and create_2d_data
def perform_noisy_actions(train_set_noisy, train_set_2d_org, camera_frame, rcams, predict_14, percent):
  #validate(train_set, train_set)
  print('Performing noisy actions now..')
  train_set_3d_noisy = addNoiseTo3D(train_set_noisy, percent)

  #got the 3d data and now do 2d things
  train_set_2d_noisy = du.project_to_cameras(train_set_3d_noisy, rcams)

  #there can be augmented 2d data
  train_set_2d_org.update(train_set_2d_noisy)

  # Compute normalization statistics.
  complete_train = copy.deepcopy(np.vstack(train_set_2d_org.values()))
  data_mean, data_std, dim_to_ignore, dim_to_use = du.normalization_stats(complete_train, dim=2)

  # Divide every dimension independently
  train_set_2d_org = du.normalize_data(train_set_2d_org, data_mean, data_std, dim_to_use)
  print("For noisy, mean :")
  print(data_mean)
  print("For noisy, std :")
  print(data_std)
  print('noisy actions done..')
  return train_set_2d_org, data_mean, data_std


def validate(dic1, dic2):
  if len(dic1) != len(dic2):
    print("Dictionaries sizes not equal")

  for key in dic1.keys():
    n1 = dic1[key]
    n2 = dic2[key]
    if not np.array_equal(n1,n2):
      print("Key", key, ", values different")

  print("Validated dictionaries .")