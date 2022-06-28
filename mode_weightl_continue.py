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

# dp = package_dataprocess.Nilm_classes.Nilm_dataprocess(path_csv="DATA/vicurve_name.xlsx",path_img="DATA/vicurve32x32/")
sum_data_num = np.zeros(100800)

filenames = os.listdir(r'DATA\UKData_by_hour\week1\LowFreq')
# print(filenames)


# In[3]:


boiler_data = np.zeros(100800)
fridge_data = np.zeros(100800)
light4_data = np.zeros(100800)
kettle_data = np.zeros(100800)
microwave_data = np.zeros(100800)
washerdryer_data = np.zeros(100800)
dishwasher_data = np.zeros(100800)

# In[4]:


for time in range(7 * 24):
    temp = np.zeros(100800)
    for i in filenames:
        if (i[:10] == str(1451865600 + 3600 * time)):#换周的话记得换一下时间戳喔
            name = i[11:-4]
            name_data = np.load('DATA/UKData_by_hour/week1/LowFreq/' + i)
            name_data = np.array(name_data)
            # print(name_data.shape)
            if len(name_data) > 600:
                name_data = name_data[:600]
            elif len(name_data) < 600:
                name_data = np.pad(name_data, (0, 600 - len(name_data)), 'constant', constant_values=(0, 0))
            name_data = np.where(name_data > 3, name_data, 0)  # 底噪3w

            if (name == 'Fridge freezer0'):
                fridge_data[time * 600:(time + 1) * 600] = name_data
            if (name == 'Light4'):
                light4_data[time * 600:(time + 1) * 600] = name_data
            if (name == 'Kettle0'):
                kettle_data[time * 600:(time + 1) * 600] = name_data
            if (name == 'Microwave0'):
                microwave_data[time * 600:(time + 1) * 600] = name_data
            if (name == 'Washer dryer0'):
                washerdryer_data[time * 600:(time + 1) * 600] = name_data
            if (name == 'Dish washer0'):
                dishwasher_data[time * 600:(time + 1) * 600] = name_data
            if (name == 'Boiler0'):
                boiler_data[time * 600:(time + 1) * 600] = name_data
            temp[time * 600:(time + 1) * 600] = name_data
            sum_data_num = sum_data_num + temp
            # print(sum_data_num.shape)
            # sum_data_num = sum_data_num[time*600:(time + 1) * 600] + name_data

# In[5]:


dp = package_dataprocess.Nilm_classes.Nilm_dataprocess(path_csv="DATA/UKData_by_hour/week1/vicurve_name.xlsx",
                                                       path_img="DATA/UKData_by_hour/week1/vicurve32x32/",
                                                       dimension = 2)

# In[7]:


fridge_data = fridge_data.reshape(fridge_data.shape[0], 1)
light4_data = light4_data.reshape(light4_data.shape[0], 1)
kettle_data = kettle_data.reshape(kettle_data.shape[0], 1)
microwave_data = microwave_data.reshape(microwave_data.shape[0], 1)
washerdryer_data = washerdryer_data.reshape(washerdryer_data.shape[0], 1)
dishwasher_data = dishwasher_data.reshape(dishwasher_data.shape[0], 1)
boiler_data = boiler_data.reshape(boiler_data.shape[0], 1)

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


fridge_labels = dp.create_label(fridge_data, 0, 15)
boiler_labels = dp.create_label(boiler_data, 1, 15)
#light4_labels = dp.create_label(light4_data, 1, 15)
#boiler_labels = dp.create_label(boiler_data, 2, 15)
sum_label = fridge_labels + boiler_labels
#sum_label = fridge_labels + light4_labels + boiler_labels

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


train_gen = dp.generator_mix(
    data=sum_data,
    label=sum_label,
    min_index=20001,
    max_index=100800)

val_gen = dp.generator_mix(
    label=sum_label,
    data=sum_data,
    min_index=0,
    max_index=20000)
test_gen = dp.generator_mix(
    data=sum_data,
    label=sum_label,
    min_index=95001,
    max_index=100800)

train_steps = (100800-20001 )// dp.batch_size
val_steps = (20000) // dp.batch_size
test_steps = (100800 - 95001 - dp.lookback) // dp.batch_size

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

from keras.models import load_model
def my_lossfun(y_true, y_pred):
    y_pred = tf.math.log(y_pred/(1-y_pred))
    #tf.print(y_pred)
    #tf.print(y_true)
    return tf.nn.weighted_cross_entropy_with_logits(labels = y_true, logits = y_pred, pos_weight = tf.constant([1.6452, 1.9641]), name=None)
model = load_model('model_Fri_Boil.hdf5', compile=False)
model.compile(loss = my_lossfun,optimizer='rmsprop', metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=[0.5, 0.5])])
filepath = 'model_Fri_Boil_continue.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                               verbose=1,
                                               monitor='val_loss',
                                               save_weights_only=False,
                                               mode='min',
                                               save_best_only=True,
                                               factor=0.1,
                                               patience=3,
                                               epsilon=1e-4)

history = model.fit_generator(train_gen,
                              steps_per_epoch=train_steps,
                              epochs=5,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              callbacks=[checkpointer])
model.summary()
