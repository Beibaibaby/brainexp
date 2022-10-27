import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.utils.vis_utils import plot_model
import datetime
import sklearn
import tensorflow as tf
print('GPU name', tf.config.experimental.list_physical_devices('GPU'))

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

all_subject_fmri_features=np.load('../all_subject_fmri_features.npy')
all_subject_ecg_features=np.load('../all_subject_ecg_features.npy')
maskelement=all_subject_fmri_features[0][0]
mask=np.nonzero(maskelement)
print("masking")
#print(mask)

print("stop masking")
#print(maskelement[mask])
all_subject_fmri_features=all_subject_fmri_features[:,:,mask[0],mask[1],mask[2]]
#print(all_subject_fmri_features)

import matplotlib.pyplot as plt
_=plt.hist(all_subject_fmri_features.flatten(),bins='auto')

plt.show()
all_subject_fmri_features=NormalizeData(all_subject_fmri_features)

_ =plt.hist(all_subject_fmri_features.flatten(),bins='auto')
plt.show()
all_subject_ecg_features=NormalizeData(all_subject_ecg_features)
_ =plt.hist(all_subject_ecg_features.flatten(),bins='auto')
plt.show()
print('fmri_mean')
print(np.mean(all_subject_fmri_features))
print('fmri_std')
print(np.std(all_subject_fmri_features))
print('ecg_mean')
print(np.mean(all_subject_ecg_features))
print('ecg_std')
print(np.std(all_subject_ecg_features))

num_sample=all_subject_fmri_features.shape[0]
num_time_step=all_subject_fmri_features.shape[1]
img_shape=all_subject_fmri_features.shape[2:]
#img_size=all_subject_fmri_features.shape[2]*all_subject_fmri_features.shape[3]*all_subject_fmri_features.shape[4]
img_size=mask[0].size
batch_size=2

print(np.max(all_subject_fmri_features))
print(np.min(all_subject_fmri_features))

import sklearn.model_selection

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(all_subject_fmri_features, all_subject_ecg_features, test_size=0.2)
x_train_for_l=x_train.reshape(*x_train.shape[:-3], -1)
print(x_train_for_l.shape)
x_train_for_l=x_train.reshape(*x_train.shape[:-3], -1)

print('Build model...')
model = Sequential()
#model.add(Embedding(max_features, 9))

#maskelement[mask]
#model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(input_shape=(mask[0].size))))
#model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(input_shape=(all_subject_fmri_features.shape[2],all_subject_fmri_features.shape[3],all_subject_fmri_features.shape[4]))))
#model.add(tf.keras.layers.Flatten(input_shape=(all_subject_fmri_features.shape[2],all_subject_fmri_features.shape[3],all_subject_fmri_features.shape[4])))
#model.add(tf.keras.layers.Conv1D(filters=num_smaples, kernel_size=3, activation='relu', input_shape=(len, 6)))

model.add(tf.keras.layers.LSTM(256, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(32, activation='relu',return_sequences=True,name='latent'))

#model.add(tf.keras.layers.RepeatVector(num_time_step))
model.add(tf.keras.layers.LSTM(64, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu',return_sequences=True))

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(img_size)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape(img_shape)))
model.build(x_train.shape)

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])
model.summary()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print('Train...')
model.fit(x_train, x_train,
          batch_size=batch_size,
          epochs=150,callbacks=[tensorboard_callback])

model.save("my_model")
score, acc = model.evaluate(x_test, x_test,
                            batch_size=batch_size)

print('MSE:', score)
print('Test accuracy:', acc)



# Train the model using the training sets




from keras.models import Model

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('latent').output)
intermediate_output_test = intermediate_layer_model.predict(x_test)
intermediate_output_train = intermediate_layer_model.predict(x_train)

print(intermediate_output_train.shape)

model2 = Sequential()
#model.add(Embedding(max_features, 9))

#model2.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(input_shape=(all_subject_fmri_features.shape[2],all_subject_fmri_features.shape[3],all_subject_fmri_features.shape[4]))))
#model.add(tf.keras.layers.Flatten(input_shape=(all_subject_fmri_features.shape[2],all_subject_fmri_features.shape[3],all_subject_fmri_features.shape[4])))
#model.add(tf.keras.layers.Conv1D(filters=num_smaples, kernel_size=3, activation='relu', input_shape=(len, 6)))

model2.add(tf.keras.layers.LSTM(16, activation='relu',return_sequences=True))
model2.add(tf.keras.layers.LSTM(8, activation='relu',return_sequences=True))
model2.add(tf.keras.layers.LSTM(4, activation='relu',return_sequences=True))
model2.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model2.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

print('Train...')
model2.fit(intermediate_output_train, y_train,
          batch_size=batch_size,
          epochs=150,callbacks=[tensorboard_callback])



model2.save("my_model_2")
score, acc = model2.evaluate(intermediate_output_test, y_test,
                            batch_size=batch_size)

print('MSE:', score)
print('Test accuracy:', acc)