#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:55:40 2022

@author: 99259
"""
import tensorflow as tf
import package_dataprocess.Nilm_classes
import numpy as np
import keras
# from nilm_ukdale import preprocess
from keras import Model
from keras.models import Sequential
import os
import gc
import pandas as pd
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import PIL.ImageOps
from keras import Model
from keras import layers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling3D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import GRU, TimeDistributed, Bidirectional
from keras.layers import Embedding
import matplotlib.pyplot as plt

# In[2]:


# 电器id定义
fridge_id = 0
tv_id = 1
kettle_id = 2
microwave_id = 3
washerdryer_id = 4

# 电器开启状态阈值
fridge_threshold = 20
tv_threshold = 20
kettle_threshold = 20
microwave_threshold = 20
washerdryer_threshold = 20

# dp = package_dataprocess.Nilm_classes.Nilm_dataprocess(path_csv="DATA/vicurve_name.xlsx",path_img="DATA/vicurve32x32/")
sum_data_num = np.zeros(100800)

filenames = os.listdir(r'DATA\UKData_by_hour\week1\LowFreq')
# print(filenames)


# In[3]:


television_data = np.zeros(100800)
fridge_data = np.zeros(100800)
microwave_data = np.zeros(100800)
washerdryer_data = np.zeros(100800)
kettle_data = np.zeros(100800)
# In[4]:


for time in range(7 * 24):
    temp = np.zeros(100800)
    for i in filenames:
        if (i[:10] == str(1451865600 + 3600 * time)):
            name = i[11:-4]
            name_data = np.load('DATA/UKData_by_hour/week1/LowFreq/' + i)
            name_data = np.array(name_data)
            # print(name_data.shape)
            if len(name_data) > 600:
                name_data = name_data[:600]
            elif len(name_data) < 600:
                name_data = np.pad(name_data, (0, 600 - len(name_data)), 'constant', constant_values=(0, 0))
            name_data = np.where(name_data > 3, name_data, 0)  # 底噪3w

            if (name == 'Television0'):
                television_data[time * 600:(time + 1) * 600] = name_data
                print("ok1")
            if (name == 'Fridge freezer0'):
                fridge_data[time * 600:(time + 1) * 600] = name_data
                print("ok2")
            if (name == 'Microwave0'):
                microwave_data[time * 600:(time + 1) * 600] = name_data
                print("ok3")
            if (name == 'Washer dryer0'):
                washerdryer_data[time * 600:(time + 1) * 600] = name_data
                print("ok4")
            if (name == 'Kettle0'):
                kettle_data[time * 600:(time + 1) * 600] = name_data
                print("ok5")
            temp[time * 600:(time + 1) * 600] = name_data
            sum_data_num = sum_data_num + temp
            # print(sum_data_num.shape)
            # sum_data_num = sum_data_num[time*600:(time + 1) * 600] + name_data

# In[5]:


dp = package_dataprocess.Nilm_classes.Nilm_dataprocess(path_csv="DATA/UKData_by_hour/week1/vicurve_name.xlsx",
                                                       path_img="DATA/UKData_by_hour/week1/vicurve32x32/")

# In[6]:



# In[7]:


television_data = television_data.reshape(television_data.shape[0], 1)
fridge_data = fridge_data.reshape(fridge_data.shape[0], 1)
microwave_data = microwave_data.reshape(microwave_data.shape[0], 1)
washerdryer_data = washerdryer_data.reshape(washerdryer_data.shape[0], 1)
kettle_data = kettle_data.reshape(kettle_data.shape[0], 1)
# fridge_data = np.load('DATA/UKData_2/Fridge freezer0.npy')
# fridge_data = fridge_data[:100800]
# fridge_data = np.where(fridge_data > dp.air_threshold,fridge_data,0)
#
# television_data = np.load('DATA/UKData_2/Television0.npy')
# television_data = television_data[:100800]
# television_data = np.where(television_data > dp.air_threshold, television_data,0)
#
# kettle_data = np.load('DATA/UKData_2/Kettle0.npy')
# kettle_data = kettle_data[:100800]
# kettle_data = np.where(kettle_data > dp.air_threshold, kettle_data,0)
#
# microwave_data = np.load('DATA/UKData_2/Microwave0.npy')
# microwave_data = microwave_data[:100800]
# microwave_data = np.where(microwave_data > dp.air_threshold, microwave_data,0)
#
# washerdryer_data = np.load('DATA/UKData_2/Washer dryer0.npy')
# washerdryer_data = washerdryer_data[:100800]
# washerdryer_data = np.where(washerdryer_data > dp.air_threshold, washerdryer_data,0)
#
# Fan_data = np.load('DATA/UKData_2/Fan0.npy')
# Fan_data = Fan_data[:100800]
# Fan_data = np.where(fridge_data > dp.air_threshold,Fan_data,0)
#
#

# In[8]:


tv_labels = dp.create_label(television_data, 2, tv_threshold)
fridge_labels = dp.create_label(fridge_data, 0, 20)
microwave_labels = dp.create_label(microwave_data, 1, microwave_threshold)
washerdryer_labels = dp.create_label(washerdryer_data, 3, washerdryer_threshold)
kettle_labels = dp.create_label(kettle_data, 4, 10)
sum_label = tv_labels + fridge_labels + microwave_labels + washerdryer_labels + kettle_labels
print(max(sum_data_num))
mean = sum_data_num[:100800].mean(axis=0)
sum_data = sum_data_num - mean
std = sum_data[:100800].std(axis=0)
sum_data /= std

# In[9]:


sum_data = np.array(sum_data)
print(sum_data.shape)

sum_data = sum_data.reshape(sum_data.shape[0], 1)
print(sum_data.shape)

# In[10]:


train_gen = dp.generator_sequence(
    data=sum_data,
    label=sum_label,
    min_index=0,
    max_index=89984)

val_gen = dp.generator_sequence(
    label=sum_label,
    data=sum_data,
    min_index=89985,
    max_index=100480)
test_gen = dp.generator_sequence(
    data=sum_data,
    label=sum_label,
    min_index=100481,
    max_index=100800)

train_steps = 10000 // dp.batch_size
val_steps = (100480 - 89985) // dp.batch_size
test_steps = (100800 - 100481 - dp.lookback) // dp.batch_size

# In[11]:


from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras.layers import InputLayer


# In[12]:


def create_cnn():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, 32, 32, 1)))
    model.add(TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(layers.Bidirectional(layers.GRU(32, dropout=0.2)))
    model.add(layers.Dense(16, activation='relu'))
    # model.add(TimeDistributed(Embedding(input_dim=2222,output_dim=32)))
    # model.add(layers.Dense(32,activation='relu'))
    # return our model
    return model


# In[13]:


model = Sequential()
model.add(InputLayer(input_shape=(None, 1)))
model.add(layers.Bidirectional(layers.GRU(64, dropout=0.2)))
model.add(layers.Dense(16, activation='relu'))
model.add(Dense(5, activation="sigmoid"))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# In[14]:


history = model.fit_generator(train_gen,
                              steps_per_epoch=train_steps,
                              epochs=10,
                              validation_data=val_gen,
                              validation_steps=val_steps,)
model.summary()

# In[13]:

'''

# In[8]:


mymodel=tf.keras.models.load_model('CNNRNNMODEL.hdf5')


# In[9]:


test_loss, test_acc = mymodel.evaluate_generator(test_gen,steps=test_steps)
print('test acc : ', test_acc)


# #keras.callbacks.ModelCheckpoint('G:\model', monitor='acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)
# model = Sequential()
# 
# model.add(TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu',input_shape=(None,32,32,1))))
# #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(2,2))))
# model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
# model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(2,2))))
# model.add(TimeDistributed(Flatten()))
# #model.add(TimeDistributed(Dense(64, activation='relu')))
# #model.add(TimeDistributed(Embedding(input_dim=2222,output_dim=32)))
# model.add(layers.Bidirectional(
#     layers.GRU(64,dropout=0.1)))
# model.add(layers.Dense(5,activation='sigmoid'))
# #model.build(input_shape=(None,32,32,1))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
# #model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',metrics=['acc'])
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=train_steps,
#                               epochs=7,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)
# model.summary()

# In[ ]:


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
# plt.legend()
plt.show()

plt.clf() #清空图表

acc_values = history.history['acc']
val_acc_values  = history.history['val_acc']

plt.plot(epochs,acc_values,'bo',label='Training acc') #bo是蓝色圆点
plt.plot(epochs,val_acc_values,'b',label='Validation acc') #b是蓝色实线
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate_generator(test_gen,steps=test_steps)
print('test acc : ', test_acc)


# In[ ]:


'''

