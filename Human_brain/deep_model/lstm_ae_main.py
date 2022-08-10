import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector


import tensorflow as tf


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


all_subject_fmri_features=np.load('../all_subject_fmri_features.npy')
all_subject_fmri_features=NormalizeData(all_subject_fmri_features)
num_sample=all_subject_fmri_features.shape[0]
num_time_step=all_subject_fmri_features.shape[1]
img_shape=all_subject_fmri_features.shape[2:]
img_size=all_subject_fmri_features.shape[2]*all_subject_fmri_features.shape[3]*all_subject_fmri_features.shape[4]
batch_size=2
print('Build model...')
model = Sequential()
#model.add(Embedding(max_features, 9))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(input_shape=(all_subject_fmri_features.shape[2],all_subject_fmri_features.shape[3],all_subject_fmri_features.shape[4]))))
#model.add(tf.keras.layers.Flatten(input_shape=(all_subject_fmri_features.shape[2],all_subject_fmri_features.shape[3],all_subject_fmri_features.shape[4])))
#model.add(tf.keras.layers.Conv1D(filters=num_smaples, kernel_size=3, activation='relu', input_shape=(len, 6)))
model.add(tf.keras.layers.LSTM(128, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu',return_sequences=True))
#model.add(tf.keras.layers.RepeatVector(num_time_step))
model.add(tf.keras.layers.LSTM(64, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu',return_sequences=True))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(img_size)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape(img_shape)))

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

print('Train...')
model.fit(all_subject_fmri_features, all_subject_fmri_features,
          batch_size=batch_size,
          epochs=50)
#score, acc = model.evaluate(x_test, y_test,batch_size=batch_size)
#print('MSE:', score)
#print('Test accuracy:', acc)


