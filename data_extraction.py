import mediapipe as mp
import cv2
import os
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                          model_complexity = 2,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

#Extracts 33 keypoints from one frame

def extract_keypoints_per_frame(image_pose):
    keypoints_per_frame=[]
    original_image = image_pose.copy()
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    resultant = pose.process(image_in_RGB)
    if resultant.pose_landmarks:
      for landmark in resultant.pose_landmarks.landmark:
        keypoints_per_frame.append([landmark.x, landmark.y, landmark.z])
    return keypoints_per_frame


#Extracts all keypoints per frame in a video
def extract_keypoints_per_video(video_path):
  total_keypoints=[]
  video = cv2.VideoCapture(video_path)
  while video.isOpened():
    ret, frame = video.read()
    if not ret:
      break
    kpoints=extract_keypoints_per_frame(frame)
    if len(kpoints)==33:
      total_keypoints.append(kpoints)
  video.release()
  cv2.destroyAllWindows()
  return total_keypoints


def extract_dataset_directdir(datadir,datadict,part):
  files = os.listdir(datadir)
  print('Downloading total data: ',len(files), 'from', datadir)
  for fl in files:
    full_path = os.path.join(datadir,fl)
    kp = extract_keypoints_per_video(full_path)
    datadict['keypoints'].append(kp)
    datadict['class'].append(datadir.split('/')[-1])
    datadict['datapart'].append(part)
    print('dataset size:',len(datadict['keypoints']))
  return datadict



train_datadir='D://dataset//trimmed//train'
val_datadir='D://dataset//trimmed//val'
ideal_datadir = 'D://dataset//trimmed//ideal//Manual_Annotations//Idle_Video_clips'
random_datadir='D://dataset//trimmed//random//Manual_Annotations'
all_data=[train_datadir,val_datadir,ideal_datadir,random_datadir]


# Calls all the required functions on all data directories
# Returns master dictionary

def master_dataset(all_data=all_data):
  dataset={'keypoints':[],'class':[],'datapart':[]}
  #print(type(all_data))
  for dir in all_data:
    if 'train' in dir:
      part = 'train'
    elif 'val' in dir:
      part = 'val'
    else:
      part = 'random'
    sub_dir=os.listdir(dir)
    for sub in sub_dir:
      full_path = os.path.join(dir,sub)
      if full_path != 'D://dataset//Manual_Annotations//Idle_Video_clips':
        extract_dataset_directdir(full_path, dataset, part)
  return dataset



# Calls all the required functions on all data directories
# Returns master dictionary

def master_dataset(all_data=all_data):
  dataset={'keypoints':[],'class':[],'datapart':[]}
  #print(type(all_data))
  for dir in all_data:
    if 'train' in dir:
      part = 'train'
    elif 'val' in dir:
      part = 'val'
    else:
      part = 'random'
    sub_dir=os.listdir(dir)
    for sub in sub_dir:
      full_path = os.path.join(dir,sub)
      if full_path != 'D://dataset//Manual_Annotations//Idle_Video_clips':
        extract_dataset_directdir(full_path, dataset, part)
  return dataset



# Converting dictionary to csv for preserverance and future use

import pandas as pd
df = pd.DataFrame(dt)
print(df.shape)
df.to_csv('D://masterdataset_pose_keypoints.csv',index=False)
