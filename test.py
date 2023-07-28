# This cell contains all the helping functions required for evaluation.
# Nothing has to be changed here just run this cell.



# Installing mediapipe
!pip install -q mediapipe==0.10.0

# Importing required libraries
import mediapipe as mp
import cv2
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                          model_complexity = 2,
                          enable_segmentation=False,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

classes = ['mountain climber (exercise)', 'jumping jacks', 'lunge','push up', 'squat', 'random']


# Extracts keypoints from one frame
def extract_keypoints_per_frame(image_pose):
    keypoints_per_frame=[]
    original_image = image_pose.copy()
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    resultant = pose.process(image_in_RGB)
    if resultant.pose_landmarks:
      for landmark in resultant.pose_landmarks.landmark:
        keypoints_per_frame.append([landmark.x, landmark.y, landmark.z])
    return keypoints_per_frame

# Extracts keypoints from one video
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

# Extracts dataset dictionary from a direct path to the folder holding videos
def extract_dataset_directdir(datadir, datadict):
  files = os.listdir(datadir)
  print('Downloading total data: ',len(files), 'from', datadir)
  for fl in files:
    full_path = os.path.join(datadir,fl)
    kp = extract_keypoints_per_video(full_path)
    datadict['keypoints'].append(kp)
    datadict['class'].append(datadir.split('/')[-1])
    print('dataset size:',len(datadict['keypoints']))
  return datadict

# Generates the master dictionary containing keypoints and class informations
def master_dataset(folder_path):
  dataset={'keypoints':[],'class':[]}
  sub_dir=os.listdir(folder_path)
  for sub in sub_dir:
    full_path = os.path.join(folder_path,sub)
    extract_dataset_directdir(full_path, dataset)
  return dataset

# Generates a dataframe from the master dictionary for better access of the data
def generate_df(folder_path):
  dt = master_dataset(folder_path)
  df = pd.DataFrame(dt)
  return df

# Processes class names in df
def check_class(df):
  for n, data in enumerate(df['class']):
    if data not in classes:
      df['class'][n]='random'
  return df

# Generates frames of shape (batch, 50, 99) and lables of shape(batch, 6)
def data_generator_for_test(df):
  frames = 50
  overlap = 15
  max_frames = max(len(seq) for seq in df['keypoints'])
  stride = frames - overlap
  num_segments_train = (max_frames - frames) // (frames - overlap) + 1
  data = np.zeros((len(df)*num_segments_train, frames, 33, 3),dtype=np.float16)
  labels = np.repeat(df['class'], num_segments_train)  # Repeat labels based on the number of segments
  for i, (seq, label) in enumerate(zip(df['keypoints'],df['class'])):
    seq_len = len(seq)
    for j in range(num_segments_train):
        start_idx = j * stride
        end_idx = start_idx + frames
        if end_idx <= seq_len:
            data[i * num_segments_train + j] = np.array(seq[start_idx:end_idx])
        else:
            padded_seq = np.pad(seq, ((0, end_idx - seq_len), (0, 0), (0, 0)), mode='constant')
            data[i * num_segments_train + j] = padded_seq[:frames]
  data = data.reshape(-1, 50, 99)
  labels = encoder.fit_transform(labels.values.reshape(-1, 1)).toarray().astype(np.float16)
  return data, labels

# Loads the model and predicts the y' after passing data and labels and prints the metrices
def model_prediction( data, label, model_h5='/content/model2.h5', model_json='/content/model2.json'):
  with open('model2.json', 'r') as json_file:
    loaded_model_json = json_file.read()
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights('model2.h5')
  predictions = loaded_model.predict(data)
  predicted_labels = np.argmax(predictions, axis=1)
  true_labels = np.argmax(label, axis=1)

  accuracy = accuracy_score(true_labels, predicted_labels) * 100
  loss = log_loss(labels, predictions)
  precision = precision_score(true_labels, predicted_labels, average='macro')* 100
  recall = recall_score(true_labels, predicted_labels, average='macro')* 100
  f1 = f1_score(true_labels, predicted_labels, average='macro')* 100

  print("Accuracy: %.2f%%" % accuracy)
  print("Loss: %.4f" % loss)
  print("Precision:", precision)
  print("Recall:", recall)
  print("F1 Score:", f1)

# Main function to run all the functions from here.
def model_evaluation(folder):
  df = generate_df(folder)
  df = check_class(df)
  data, labels = data_generator_for_test(df)
  print('number of frames generated :', data.shape[0], 'with classes:', labels.shape[1])
  model_prediction(data, labels)

# Calling the main evaluation function
# running of this cell may take some time as it creates data using mediapipe
# change the folder path here
model_evaluation('<path_to_test_folder>')
