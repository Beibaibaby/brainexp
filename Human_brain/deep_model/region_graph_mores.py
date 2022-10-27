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

def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

all_subject_fmri_features_pre=np.load('../parcellation/all_subject_fmri_features_region.npy')
all_subject_ecg_features=np.load('../parcellation/all_subject_ecg_features_region.npy')
all_subject_fmri_features=all_subject_fmri_features_pre

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

pss=np.asarray([0.9,1.5,1.9,1.6,1.5,0.8,2.5,1.4,1.1,2,0.3,0.5,0.9,0.7,2.5,1.7,0.6,1.1,1.9,0.8,0.8,0.8])
pss=NormalizeData(pss)
print(pss)



all_subject_fmri_features=[]
print(all_subject_fmri_features_pre.shape)
for i in range(all_subject_fmri_features_pre.shape[0]):
    this_fc_ts=[]
    for j in range(all_subject_fmri_features_pre.shape[1]):
        this_fc_ts.append(np.append(upper_tri_indexing(all_subject_fmri_features_pre[i][j]),all_subject_ecg_features[i][j]))
    all_subject_fmri_features.append(this_fc_ts)
all_subject_fmri_features = np.asarray(all_subject_fmri_features)
print(all_subject_fmri_features.shape)
all_subject_fmri_features_tep=[]
pss_tep=[]
for i in range(all_subject_fmri_features.shape[0]):
    for j in range(10):
        this=all_subject_fmri_features[i,5*j:(5*j+5)]
        all_subject_fmri_features_tep.append(this)
        pss_tep.append(pss[i])
all_subject_fmri_features=np.asarray(all_subject_fmri_features_tep)
pss=np.asarray(pss_tep)

num_sample=all_subject_fmri_features.shape[0]
num_time_step=all_subject_fmri_features.shape[1]
img_shape=all_subject_fmri_features.shape[2:]
#img_size=all_subject_fmri_features.shape[2]*all_subject_fmri_features.shape[3]*all_subject_fmri_features.shape[4]

batch_size=2

print(np.max(all_subject_fmri_features))
print(np.min(all_subject_fmri_features))



import sklearn.model_selection

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(all_subject_fmri_features,pss , test_size=0.2)
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

#model.add(tf.keras.layers.LSTM(1024, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(32, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(16, activation='relu',return_sequences=True))
model.add(tf.keras.layers.LSTM(1, activation='relu',return_sequences=False))
#model.add(tf.keras.layers.RepeatVector(num_time_step))
#model.add(tf.keras.layers.LSTM(64, activation='relu',return_sequences=True))
#model.add(tf.keras.layers.LSTM(128, activation='relu',return_sequences=True))
#model.add(tf.keras.layers.LSTM(256, activation='relu',return_sequences=True))

#model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(img_size)))
#model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape(img_shape)))
model.build(x_train.shape)

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
              metrics=['mse'])
model.summary()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=150,callbacks=[tensorboard_callback])

model.save("my_model")
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('MSE:', score)
print('Test accuracy:', acc)



# Train the model using the training sets

