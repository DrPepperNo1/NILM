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

from keras.layers import Layer
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
import tensorflow as tf

sum_data_num = np.zeros(100800)

filenames = os.listdir(r'DATA\UKData_by_hour\week2\LowFreq')

boiler_data = np.zeros(100800)
fridge_data = np.zeros(100800)
light4_data = np.zeros(100800)
kettle_data = np.zeros(100800)
microwave_data = np.zeros(100800)
washerdryer_data = np.zeros(100800)
dishwasher_data = np.zeros(100800)

for time in range(7 * 24):
    temp = np.zeros(100800)
    for i in filenames:
        if (i[:10] == str(1452470400 + 3600 * time)):
            name = i[11:-4]
            name_data = np.load('DATA/UKData_by_hour/week2/LowFreq/' + i)
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

dp = package_dataprocess.Nilm_classes.Nilm_dataprocess(path_csv="DATA/UKData_by_hour/week2/vicurve_name.xlsx",
                                                       path_img="DATA/UKData_by_hour/week2/vicurve32x32/",
                                                       dimension = 3)

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
light4_labels = dp.create_label(light4_data, 1, 15)
boiler_labels = dp.create_label(boiler_data, 2, 15)

sum_label = fridge_labels + light4_labels + boiler_labels

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
    min_index=0,
    max_index=80000)

val_gen = dp.generator_mix(
    label=sum_label,
    data=sum_data,
    min_index=80001,
    max_index=100800)
test_gen = dp.generator_mix(
    data=sum_data,
    label=sum_label,
    min_index=100481,
    max_index=100800)

train_steps = 80000 // dp.batch_size
val_steps = (100800 - 80001) // dp.batch_size
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


class PrintLayer(Layer):
	#初始化方法，不须改变
    def __init__(self, getname = 'defaultname', **kwargs):
        super(PrintLayer, self).__init__(**kwargs)
        self.nameHH = getname
	#调用该层时执行的方法
    def call(self, x):
        x = tf.compat.v1.Print(x,[x],message=self.nameHH +" is: ",summarize=65536, name = self.nameHH)
        #调用tf的Print方法打印tensor方法，第一个参数为输入的x，第二个参数为要输出的参数，summarize参数为输出的元素个数。
        return x;
        #一定要返回tf.Print()函数返回的变量，不要直接使用传入的变量。

PrintLayer()
# In[12]:
def create_cnn():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, 32, 32, 1)))
    model.add(TimeDistributed(layers.Conv2D(64, (3, 3))))
    model.add(PrintLayer(getname='Conv0'))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(PrintLayer(getname='BN0'))
    model.add(TimeDistributed(Activation('relu')))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
    # model.add(TimeDistributed(Conv2D(64, (3, 3))))
    model.add(TimeDistributed(layers.Conv2D(16, (3, 3))))
    model.add(PrintLayer(getname='Conv1'))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(PrintLayer(getname='BN1'))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(32)))
    model.add(PrintLayer(getname='Dense0'))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(PrintLayer(getname='BN2'))
    model.add(TimeDistributed(Activation('relu')))
    model.add(layers.Bidirectional(layers.GRU(32, dropout=0.2)))
    model.add(layers.Dense(32))
    model.add(PrintLayer(getname='Dense1'))
    model.add(BatchNormalization())
    model.add(PrintLayer(getname='BN3'))
    model.add(Activation('relu'))
    # model.add(TimeDistributed(Embedding(input_dim=2222,output_dim=32)))
    # model.add(layers.Dense(32))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # return our model
    return model


'''
def create_cnn():
    model = Sequential()
    model.add(InputLayer(input_shape=(None,32,32,1)))
    model.add(TimeDistributed(layers.Conv2D(32, (3, 3))))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(2,2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3))))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(2,2))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(32)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(layers.Bidirectional(layers.GRU(32, dropout=0.2)))
    model.add(layers.Dense(16))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(TimeDistributed(Embedding(input_dim=2222,output_dim=32)))
    #model.add(layers.Dense(32))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    # return our model
    return model

'''


# In[13]:


def create_seq():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, 1)))
    model.add(layers.Bidirectional(
        layers.GRU(32, dropout=0.2)))
    model.add(layers.Dense(32))
    model.add(PrintLayer(getname='Dense00'))
    model.add(BatchNormalization())
    model.add(PrintLayer(getname='BN00'))
    model.add(Activation('relu'))
    return model


# In[14]:


branch_cnn = create_cnn()
branch_seq = create_seq()
combined = concatenate([branch_seq.output, branch_cnn.output])
x = Dense(32, activation='relu')(combined)
# x = TimeDistributed(Dense(64, activation='relu'))(x)
# x = layers.GRU(32,dropout=0.2)(x)
x = Dense(3)(x)
x = Activation('sigmoid')(x)
model = Model(inputs=[branch_seq.input, branch_cnn.input], outputs=x)

posweight = tf.constant([1.645, 140.176 , 143.294, 5.534, 27.356, 32.949, 1.964,1])
def my_lossfun(y_true, y_pred):
    y_pred = tf.math.log(y_pred/(1-y_pred))
    #tf.print(y_pred)
    #tf.print(y_true)
    return tf.nn.weighted_cross_entropy_with_logits(labels = y_true, logits = y_pred, pos_weight = tf.constant([1.6452, 1.0779, 1.9641]), name=None)
# In[13]:


filepath = 'model_Fri_light_Boil.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                               verbose=1,
                                               monitor='val_loss',
                                               save_weights_only=False,
                                               mode='min',
                                               save_best_only=True,
                                               factor=0.1,
                                               patience=3,
                                               epsilon=1e-4)

model.compile(optimizer='rmsprop', loss=my_lossfun, metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=[0.5, 0.5, 0.5])])

# In[14]:


history = model.fit_generator(train_gen,
                              steps_per_epoch=train_steps,
                              epochs=5,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              callbacks=[checkpointer])
model.summary()

# In[13]:


model.save("mymodel_Fri_light_Boil.h5")
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

