import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv1D, Dense, GRU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback



df=pd.read_csv('D://masterdataset_pose_keypoints.csv')

#converting string to list
df['keypoints']=df['keypoints'].apply(lambda x:ast.literal_eval(x))

#extracting number of frames with data per video
df['frames']=df['keypoints'].apply(lambda x:len(x))

grp = df.groupby('datapart')

random_df = df[df['datapart'] == 'random']
df = df[df['datapart'] != 'random']

random_df['class']='random'
df=df.append(random_df)

frames = 50
overlap = 15
max_frames = max(len(seq) for seq in df['keypoints'])
print(max_frames)

# Calculating the stride based on the overlap
stride = frames - overlap

# Calculating the number of segments for train_data and val_data
num_segments_train = (max_frames - frames) // (frames - overlap) + 1

# Creating empty arrays to hold the data and labels
data = np.zeros((len(df)*num_segments_train, frames, 33, 3),dtype=np.float16)
labels = np.repeat(df['class'], num_segments_train)  # Repeat labels based on the number of segments

# Filling the arrays with the keypoints data
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


# Reshaping the data into a 2D array
data_2d = data.reshape(-1, 4950)

# Splitting the data and labels into train and test sets in 80-20 ratio
train_data, test_data, train_labels, test_labels = train_test_split(data_2d, labels, test_size=0.2, random_state=42)

# Reshaping the train and test data back to 3D arrays to be sent to GRU layers
train_data = train_data.reshape(-1, 50, 99)
test_data = test_data.reshape(-1, 50, 99)


# encoding target values
encoder = OneHotEncoder()
train_labels_encoded = encoder.fit_transform(train_labels.values.reshape(-1, 1)).toarray().astype(np.float16)
test_labels_encoded = encoder.transform(test_labels.values.reshape(-1, 1)).toarray().astype(np.float16)


initial_lr = 0.001  # Initial learning rate

def lr_scheduler(epoch, lr):
    if epoch % 20 == 0 and epoch > 0:
        return lr * 0.1
    else:
        return lr

#lr_scheduler will decay LR by 1/10 after every 20 epochs

lr_schedule = LearningRateScheduler(lr_scheduler)

#compiling the model with the Adam optimizer and the custom learning rate schedule
optimizer = Adam(learning_rate=initial_lr)





# model building
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(50, 99)))
model.add(Dropout(0.5))
model.add(GRU(32))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# using a callback to print the used LR after every epoch
class LearningRateCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr  # Get the current learning rate
        print(f"Learning Rate: {lr}")

lr_callback = LearningRateCallback()

#print the model summary
model.summary()


#model training
history = model.fit(train_data, train_labels_encoded, validation_data=(test_data, test_labels_encoded), epochs=50, batch_size=32)


#saving the model
model.save('model1.h5')

# Save the model as JSON file
model_json = model.to_json()
with open('model1.json', 'w') as json_file:
    json_file.write(model_json)
