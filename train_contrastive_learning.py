import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv1D, Dense, GRU, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model


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

def make_pairs(data, lables):
    pair_data = []
    pair_lables = []
    num_classes = lables.shape[1]

    for idx_A in range(len(data)):
        current_data = data[idx_A]
        lable = lables[idx_A]

        # Randomly selecting a positive sample from the same class
        positive_indices = np.where(np.argmax(lables, axis=1) == np.argmax(lable))[0]
        positive_idx = np.random.choice(positive_indices)
        positive_data = data[positive_idx]

        # Preparing positive pair
        pair_data.append([current_data, positive_data])
        pair_lables.append(1) # symobolizing pair data from same class

        # Randomly selecting a negative sample from a different class
        negative_indices = np.where(np.argmax(lables, axis=1) != np.argmax(lable))[0]
        negative_idx = np.random.choice(negative_indices)
        negative_data = data[negative_idx]

        # Preparing negative pair
        pair_data.append([current_data, negative_data])
        pair_lables.append(0)

    return (np.array(pair_data), np.array(pair_lables))

pair_test, lable_test = make_pairs(test_data, test_labels_encoded) #data for validation

def euclidean_distance(vectors):
    # Unpacking the vectors into separate tensors
    featsA, featsB = vectors

    # Computing the sum of squared distances between the vectors
    sumSquared = tf.reduce_sum(tf.square(featsA - featsB), axis=1, keepdims=True)

    # Returning the Euclidean distance between the vectors
    return tf.sqrt(tf.maximum(sumSquared, tf.keras.backend.epsilon()))

#function to calculate contrastive loss

def contrastive_loss(y, preds, margin=1):
    # Explicitly casting the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)

    # Calculating the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = tf.square(preds)
    squaredMargin = tf.square(tf.maximum(margin - preds, 0))
    loss = tf.reduce_mean(y * squaredPreds + (1 - y) * squaredMargin)

    # Returning the computed contrastive loss to the calling function
    return loss

# creating data generator functions to be used for data loading during runtime to save RAM usage
def data_generator_train(pair_train, label_train):
    inp_1, inp_2 = pair_train[:, 0, :, :], pair_train[:, 1, :, :] # decoupling the pair inputs
    for i in range(len(pair_train)):
      yield (inp_1[i].astype(np.float16), inp_2[i].astype(np.float16)), lable_train[i].astype(np.float16)
      
train_dataset = tf.data.Dataset.from_generator(
    data_generator_train,
    output_signature=(
        ((tf.TensorSpec(shape=(50, 99), dtype=tf.float16), tf.TensorSpec(shape=(50, 99), dtype=tf.float16))),
        tf.TensorSpec(shape=(), dtype=tf.float16)
    ),
    args=(pair_train, lable_train)
)
train_dataset = train_dataset.batch(8)

def data_generator_test(pair_test, lable_test):
  inp_1, inp_2 = pair_test[:, 0, :, :], pair_test[:, 1, :, :]
  for i in range(len(pair_test)):
    yield (inp_1[i].astype(np.float16), inp_2[i].astype(np.float16)), lable_test[i].astype(np.float16)


test_dataset = tf.data.Dataset.from_generator(
    data_generator_test,
    output_signature=(
        ((tf.TensorSpec(shape=(50, 99), dtype=tf.float16), tf.TensorSpec(shape=(50, 99), dtype=tf.float16))),
        tf.TensorSpec(shape=(), dtype=tf.float16)
    ),
    args=(pair_test, lable_test)
)

test_dataset = test_dataset.batch(8)


# building siamese network
def build_siamese_model2(inputShape, embeddingDim=48):
  inputs = Input(inputShape)
  x = GRU(32,return_sequences=True)(inputs)
  x = Dropout(0.25)(x)
  x = GRU(16)(x)
  x = Dropout(0.25)(x)
  x = Dense(16)(x)
  outputs = Dense(embeddingDim)(x)
	# build the model
  model = Model(inputs, outputs)
  # return the model to the calling function
  return model

# building siamese network
input_shape = (50, 99)
keyA = Input(shape=input_shape)
keyB = Input(shape=input_shape)

featureExtractor2 = build_siamese_model2(input_shape)
featsA = featureExtractor2(keyA)
featsB = featureExtractor2(keyB)

distance = Lambda(euclidean_distance)([featsA, featsB])
model2 = Model(inputs=[keyA, keyB], outputs=distance)

model2.compile(loss=contrastive_loss, optimizer="adam")
model2.summary()

# training the model
history = model2.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=30
)


## Save the model
model2.save('model_contrastive_loss2.h5')
model_json = model2.to_json()
with open('model_contrastive_loss_2.json', 'w') as json_file:
    json_file.write(model_json)


# Extracting siamese layer from above trained model
md = loaded_model.get_layer('model')

# Testing for output shape verifiaction
input_data = np.random.rand(10,50, 99)
output_features = md.predict(input_data)

# building classifier on top of siamese network
model = Sequential()
model.add(md)
model.layers[0].trainable = False
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#training the classifier
history = model.fit(train_data, train_labels_encoded, validation_data=(test_data, test_labels_encoded), epochs=50, batch_size=32)


#saving the model
model.save('model_contrastive_loss.h5')

# Save the model as JSON file
model_json = model.to_json()
with open('model_contrastive_loss.json', 'w') as json_file:
    json_file.write(model_json)
