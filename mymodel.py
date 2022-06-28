#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:55:40 2022

@author: 99259
"""
import package_dataprocess.Nilm_classes
import numpy as np
import keras
#from nilm_ukdale import preprocess
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
from keras.layers import GRU,TimeDistributed,Bidirectional
from keras.layers import Embedding
import matplotlib.pyplot as plt


# In[2]:


#电器id定义
fridge_id = 0
tv_id = 1
kettle_id = 2
microwave_id = 3
washerdryer_id = 4

#电器开启状态阈值
fridge_threshold = 20
tv_threshold = 20
kettle_threshold = 20
microwave_threshold = 10
washerdryer_threshold = 0

dp = package_dataprocess.Nilm_classes.Nilm_dataprocess(path_csv=r"DATA/vicurve_name.xlsx",path_img=r"DATA/vicurve32x32/")

fridge_data = np.load('DATA/UKData_2/Fridge freezer0.npy')
fridge_data = fridge_data[:100800]
fridge_data = np.where(fridge_data > dp.air_threshold,fridge_data,0)

television_data = np.load('DATA/UKData_2/Television0.npy')
television_data = television_data[:100800]
television_data = np.where(television_data > dp.air_threshold, television_data,0)

kettle_data = np.load('DATA/UKData_2/Kettle0.npy')
kettle_data = kettle_data[:100800]
kettle_data = np.where(kettle_data > dp.air_threshold, kettle_data,0)

microwave_data = np.load('DATA/UKData_2/Microwave0.npy')
microwave_data = microwave_data[:100800]
microwave_data = np.where(microwave_data > dp.air_threshold, microwave_data,0)

washerdryer_data = np.load('DATA/UKData_2/Washer dryer0.npy')
washerdryer_data = washerdryer_data[:100800]
washerdryer_data = np.where(washerdryer_data > dp.air_threshold, washerdryer_data,0)

fridge_labels = dp.create_label(fridge_data,fridge_id,fridge_threshold)
tv_labels = dp.create_label(television_data,tv_id,tv_threshold)
kettle_labels = dp.create_label(kettle_data,kettle_id,kettle_threshold)
microwave_labels = dp.create_label(microwave_data,microwave_id,microwave_threshold)
washerdryer_labels = dp.create_label(washerdryer_data,washerdryer_id,washerdryer_threshold)

sum_label = fridge_labels + tv_labels + kettle_labels + microwave_labels + washerdryer_labels


train_gen = dp.generator_image(
                      label=sum_label,

                      min_index=0,
                      max_index=85000,)

val_gen = dp.generator_image(
                    label=sum_label,

                    min_index=85001,
                    max_index=95000,)
test_gen = dp.generator_image(
                    label=sum_label,
                    min_index=95001,
                    max_index=100800,)

train_steps = 85000 // dp.batch_size
val_steps = (95000 - 85001 -dp.lookback) // dp.batch_size
test_steps = (100800 - 95001 -dp.lookback) // dp.batch_size


# In[3]:


#keras.callbacks.ModelCheckpoint('G:\model', monitor='acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)
model = Sequential()

model.add(TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu',input_shape=(None,32,32,1))))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(2,2))))
model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(2,2))))
model.add(TimeDistributed(Flatten()))
#model.add(TimeDistributed(Dense(64, activation='relu')))
#model.add(TimeDistributed(Embedding(input_dim=2222,output_dim=32)))
model.add(layers.Bidirectional(
    layers.GRU(64,dropout=0.1)))
model.add(layers.Dense(5,activation='sigmoid'))
#model.build(input_shape=(None,32,32,1))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
#model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',metrics=['acc'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=train_steps,
                              epochs=7,
                              validation_data=val_gen,
                              validation_steps=val_steps)
model.summary()


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




