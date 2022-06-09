# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
import time
import numpy as np

print(tf.__version__)
vth=np.loadtxt("old-code/OR_Vth.txt",delimiter=',')
SP=np.loadtxt("old-code/OR_SP.txt",delimiter=',')
T=np.loadtxt("old-code/OR_T.txt",delimiter=',')
year=np.loadtxt("old-code/OR_year.txt",delimiter=',')
WL_ratio=np.loadtxt("old-code/OR_WL_ratio.txt",delimiter=',')
CL=np.loadtxt("old-code/OR_CL.txt",delimiter=',')
list_index=[i for i in range(1000) if i%19==0]
list_index_test=[i for i in range(1000) if i%23==0]
train_data_org=np.row_stack([vth[list_index,:],SP[list_index,:],T[list_index,:],year[list_index,:],WL_ratio[list_index,:],CL[list_index,:]])
test_data_org=np.row_stack([vth[list_index_test,:],SP[list_index_test,:],T[list_index_test,:],year[list_index_test,:],WL_ratio[list_index_test,:],CL[list_index_test,:]])
#data=np.column_stack([vth,SP,T,t,WL_ratio,CL])
#data=np.loadtxt("housing.txt",delimiter=None)
train_data=train_data_org[0:402,0:7]
train_labels=train_data_org[0:402,7]
train_labels=train_labels*10**11
test_data=test_data_org[0:100,0:7]
test_labels= test_data_org[0:100,7]
test_labels=test_labels*10**11
#order = np.argsort(np.random.random(train_labels.shape))
#train_data = train_data[order]
#train_labels = train_labels[order]
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features
print(train_data[0])  # Display sample features, notice the different scales
#import pandas as pd
#
#column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
#                'TAX', 'PTRATIO', 'B', 'LSTAT']
#
#df = pd.DataFrame(train_data, columns=column_names)
#df.head()
#print(train_labels[0:10])  # Display first 10 entries
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64,activation=tf.nn.relu ),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 50

# Store training stats

# history = model.fit(train_data, train_labels, epochs=EPOCHS,
#                     validation_split=0.2, verbose=0,
#                     callbacks=[PrintDot()])

import matplotlib.pyplot as plt


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

#plot_history(history)
start_time=time.time()
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])
end_time=time.time()
#plot_history(history)
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae))
test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions,zorder=3)
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, test_predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, test_predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, test_predictions)))
print("r^2 score:",metrics.r2_score(test_labels,test_predictions)) 
print(end_time-start_time)
plt.xlabel('True Values ')
plt.ylabel('Predictions ')
plt.axis('equal')
#plt.xlim(plt.xlim())
#plt.ylim(plt.ylim())
_ = plt.plot(test_labels, test_labels,color='red',zorder=2)
plt.grid()
plt.savefig("figNOTdeep1")
error = test_predictions - test_labels
#plt.hist(error, bins = 50)
#plt.xlabel("Prediction Error [1000$]")
#_ = plt.ylabel("Count")
