import tensorflow as tf
import package_dataprocess.Nilm_classes
import numpy as np
import keras
#from nilm_ukdale import preprocess
from keras import Model
from keras.models import Sequential
import os
ReadByHour = package_dataprocess.Nilm_classes.Nilm_dataprocess(path_csv="DATA/vicurve_name.xlsx",path_img="DATA/vicurve32x32/")
ReadByHour.ReadBuilding_byhour(building = 1, path_h5 = r'D:\ukdale.h5', path_save = 'DATA/UKData_by_hour/', start = 1455494400, week = 'week7')