# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 14:15:04 2022

@author: 99259
"""
import os
import gc
import wave
import numpy as np
#from nilmtk import DataSet
import pandas as pd
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import pandas as pd
import cv2
import io


class Nilm_vicurve:
    def __init__(self):
        pass

    def standlize_n1to1(self, data):
        data = data - np.mean(data)
        data = data / np.max(np.abs(data))
        return data

    def wavtolist(self, path_img, countsecond, counthour, countday, countweek, wave_data, sampling_rate, size_output,
                  sampwidth):
        # 打开wav文件 ，open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据。
        # f = wave.open(path_open)
        # 读取格式信息
        # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
        # params = f.getparams()
        # nchannels, sampwidth, framerate, nframes = params[:4]
        # 读取波形数据
        # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
        #print(sampwidth)
        # str_data  = f.readframes(nframes)
        # f.close()
        # 将波形数据转换成数组
        # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
        # wave_data = np.frombuffer(str_data,dtype = np.int16)
        # 将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
        # wave_data.shape = -1,2
        # 转置数据
        # wave_data = wave_data/100
        # wave_data = wave_data.T
        # wave_data[0]=standlize_n1to1(wave_data[0])
        # wave_data[1]=wave_data[1]-np.mean(wave_data[1])
        # wave_data[1]=wave_data[1]/65.0
        # 通过取样点数和取样频率计算出每个取样的时间。
        # time=np.arange(0,nframes)/framerate
        # print(params)
        # time 也是一个数组，与wave_data[0]或wave_data[1]配对形成系列点坐标
        fig = plt.figure(figsize=(size_output, size_output))
        plt.plot(wave_data[0][countsecond:countsecond + sampling_rate * 6],
                 wave_data[1][countsecond:countsecond + sampling_rate * 6], color='black')  # wave_data[0]是电压
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axis('off')
        plt.savefig(
            path_img + 'year_2016_week_' + countweek + '_' + 'day_' + str(countday) + '_hour_' + format(counthour,
                                                                                                        '02d') + '_second_' + format(
                int(countsecond / 96000), '03d') + '.jpg')
        fig.clf()
        plt.close()
        gc.collect()
        return

    def vicurve_saveas_csv(self, path_img, path_csv):
        '''
        讲文件夹内的图片名保存在.csv中

        Parameters
        ----------
        path_img : str
            DESCRIPTION.
            存储.jpg文件夹的绝对路径
        path_csv : str
            DESCRIPTION.
            .csv文件夹的绝对路径,.csv必须提前创建好
            例子 'D:\vicurve_name.csv'
        Returns
        -------
        None.

        '''
        a = []
        for filename in os.listdir(path_img):
            a.append(filename)
        f = open(path_csv, 'w')
        for line in a:
            f.write(line + '\n')
        f.close()
        return

    def wav_to_image(self, path_wav, path_img, path_csv, day, hour, week, sampling_rate, size_output):
        '''
        把.wav文件转化为.jpg储存起来

        Parameters
        ----------
        path_wav : str
            DESCRIPTION
            存放.wav的文件夹绝对路径,这个文件夹里存放一周的数据，例 E:\\vicurve_128x128\\
        path_img : str
            DESCRIPTION.
            存放.jpg的文件夹绝对路径,这个文件夹里存放一周的数据,例 E:\\vicurve_128x128\\
        day : int
            DESCRIPTION.
            指明从这一周的第几天开始,范围0~6
        hour : int
            DESCRIPTION.
            指明从第几小时开始,范围0~23
        sampling_rate : int
            DESCRIPTION.
            高频数据采样率,ukdale中为16000
        size_output : float
            DESCRIPTION.
            输出图片大小,size_output=0.64代表64x64

        Returns
        -------
        None.

        '''
        # path_save='E:\\vicurve_128x128\\' #把路径改为保存图片的路径
        filenames = os.listdir(path_wav)
        # hour=19
        # day=2
        week = '%02d' % week
        for filename in filenames:
            i = 0
            f = wave.open(path_wav + filename)
            params = f.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            str_data = f.readframes(nframes)
            wave_data = np.frombuffer(str_data, dtype=np.int16)
            wave_data.shape = -1, 2
            wave_data = wave_data / 100
            wave_data = wave_data.T
            wave_data[0] = self.standlize_n1to1(wave_data[0])
            wave_data[1] = wave_data[1] - np.mean(wave_data[1])
            wave_data[1] = wave_data[1] / 65.0
            for m in range(0, 600):
                self.wavtolist(path_img, i, hour, day, week, wave_data, sampling_rate, size_output, sampwidth)
                i += (sampling_rate * 6)
            hour += 1
            f.close()
            if hour == 24:
                hour = 0
                day += 1
        #self.vicurve_saveas_csv(path_img, path_csv)
        return


class Nilm_dataprocess(Nilm_vicurve):
    def __init__(self, path_img, path_csv,dimension = 5):
        Nilm_vicurve.__init__(self)
        self.dimension = dimension  # 电器个数
        # 电器id定义
        self.fridge_id = 0
        self.tv_id = 1
        self.kettle_id = 2
        self.microwave_id = 3
        self.washerdryer_id = 4

        # 电器开启状态阈值
        self.fridge_threshold = 20
        self.tv_threshold = 20
        self.kettle_threshold = 20
        self.microwave_threshold = 10
        self.washerdryer_threshold = 0

        # 消噪音阈值
        self.air_threshold = 5

        # 一些路径
        self.path_img = path_img
        self.path_csv = path_csv

        # 网络数据的参数
        self.lookback = 20
        self.delay = 0
        self.shuffle = False
        self.batch_size = 64

    def create_label(self, data, application_id, power_threshold):
        '''
        低频

        create_label方法创建电器运行与否的标签，标签为的形状  时间长度*电器个数
        0表示电器没有运行，1表示电器运行

        !!注意事项
        - 每次只能对一个电器创建标签
        washerdryer_data = np.load('UKData/Washer dryer0.npy')
        washerdryer_data = washerdryer_data[:1600000]
        - 把所有创建好的标签加起来
        sum_label = fridge_labels + tv_labels + kettle_labels + microwave_labels + washerdryer_labels

        parameter
        ----------
        data(list):
            一个电器的数据，如上边的washerdryer_data
        application_id(int):
            电器的序号，从0开始，序号需要对齐
        power_threshold(int):
            每个电器的阈值
        '''
        data = np.array(data)
        data = np.where(data > self.air_threshold, data, 0)
        labels = np.zeros((len(data), self.dimension))
        for i, value in enumerate(data):
            if value[0] > power_threshold:
                labels[i, application_id] = 1
        return labels
    def ReadBuilding_byhour(self, building, path_h5, path_save, start ,week):
        '''
                低频

                ReadBuilding方法读出指定房间、指定时间内所有电器的数据，按照电器，分别保存到绝对路径文件夹./UKData下.npy格式
                示例：ReadBuilding(building=1,start=1451865600,end=1452470400)

                parameter
                ----------
                building(int):
                    房间的序号
                path(str):
                    ukdale.h5的路径
                start(int):
                    起始的unix时间戳
                end(int):
                    截止的unix时间戳
                building为房间编号，path为ukdale.h5的路径

                '''
        test = DataSet(path_h5)
        for i in range(24*7):
            print(i)
            #test.set_window(start = pd.Timestamp(start+i*3600), end = pd.Timestamp(start+(i+1)*3600))  ## 2013年3月18号之后的作为数据集
            test.set_window(start=pd.to_datetime(start+i*3600, unit='s'),end=pd.to_datetime(start+(i+1)*3600, unit='s'))  ## 2013年3月18号之后的作为数据集
            #print(pd.to_datetime(1451865600, unit='s'))
            test_elec = test.buildings[building].elec
            gt = {}
            for meter in test_elec.submeters().meters:
                gen = next(meter.load(), 0)
                if(isinstance(gen,int)):
                    values = np.zeros(600)
                    index = np.zeros(3)
                else:
                    values = gen.values
                    index = gen.index
                label = meter.label()
                j = 0;
                while (1):
                    name = str(start+i*3600) +'_'+ label + str(j)
                    if name not in gt:
                        break
                    else:
                        j += 1

                if(values.shape != (0,0) and values.size > 450):
                    values = self.fullfilllist(values)

                else:
                    print(name)
                    values = np.zeros(600)

                np.save(path_save + week + '/LowFreq/' + name, values)
                gt[name] = 1
                print(values.shape, '   ', index.shape)
            del test_elec  # for循环每运行完一次要把这个对象删除
        print('读取完毕')
        return
    def fullfilllist(self, array):
        x = len(array)
        for i in range(600 - x):
            array = np.insert(array, int((i+1)*x/(600-x)), (array[int((i+1)*x/(600-x)-1)]+array[int((i+1)*x/(600-x)+1)])/2)
        return array
    def ReadBuilding(self, building, path, start, end):
        '''
        低频

        ReadBuilding方法读出指定房间、指定时间内所有电器的数据，按照电器，分别保存到绝对路径文件夹./UKData下.npy格式
        示例：ReadBuilding(building=1,start=1451865600,end=1452470400)

        parameter
        ----------
        building(int):
            房间的序号
        path(str):
            ukdale.h5的路径
        start(int):
            起始的unix时间戳
        end(int):
            截止的unix时间戳
        building为房间编号，path为ukdale.h5的路径

        '''
        test = DataSet(path)
        test.set_window(start, end)  ## 2013年3月18号之后的作为数据集
        test_elec = test.buildings[building].elec
        gt = {}
        for meter in test_elec.submeters().meters:
            gen = next(meter.load())
            values = gen.values
            index = gen.index
            label = meter.label()
            i = 0;
            while (1):
                name = label + str(i)
                if name not in gt:
                    break
                else:
                    i += 1
            np.save('UKData_2/' + name, values)
            gt[name] = 1
            print(values.shape, '   ', index.shape)
            print('saving...', name)

    def generator_mix(self, data, label, min_index, max_index, step=1):
        if max_index is None:
            pass
            # max_index = len(data) - self.delay - 1
        i = min_index + self.lookback
        while 1:
            csvfile = pd.read_excel(self.path_csv)
            if self.shuffle:
                rows = np.random.randint(min_index + self.lookback, max_index, size=self.batch_size)
            else:
                if i + self.batch_size >= max_index:
                    i = min_index + self.lookback
                rows_cnn = np.arange(i, min(i + self.batch_size, max_index))
                i += len(rows_cnn)
            samples_seq = np.zeros((len(rows_cnn),
                                    self.lookback // step,
                                    data.shape[-1]))
            samples_image = np.zeros((len(rows_cnn),
                                      self.lookback // step,
                                      32,
                                      32,
                                      1))
            targets = np.zeros((len(rows_cnn), self.dimension))
            # data_image_temp1=[]
            data_image_temp2 = []
            for j, row in enumerate(rows_cnn):
                indices_seq = range(rows_cnn[j] - self.lookback, rows_cnn[j], step)
                samples_seq[j] = data[indices_seq]
                data_image_temp2 = []
                indices = range(rows_cnn[j] - self.lookback, rows_cnn[j], step)  # row[j]对应csv的行数
                # count=0
                for lines in indices:
                    imagefilename = csvfile.iloc[lines, 0]
                    img = self.image_to_numpy(self.path_img + imagefilename)
                    # data_image_temp1.append(img)
                    # data_image_temp2.append(data_image_temp1)
                    data_image_temp2.append(img)
                    # data_image_temp1=[]
                    # count+=1
                # c=np.array(data_image_temp2).shape
                # samples_image[j,:,:,:,:]=data_image_temp2[:,:,:,:]
                samples_image[j] = data_image_temp2
                # samples_image[j].append(data_image_temp2)
                targets[j] = label[rows_cnn[j] + self.delay]
            yield [samples_seq, samples_image], targets

    def generator_sequence(self, data, label, min_index, max_index, step=1):
        if max_index is None:
            max_index = len(data) - self.delay - 1
        i = min_index + self.lookback
        while 1:
            if self.shuffle:
                rows = np.random.randint(min_index + self.lookback, max_index, size=self.batch_size)
            else:
                if i + self.batch_size >= max_index:
                    i = min_index + self.lookback
                rows = np.arange(i, min(i + self.batch_size, max_index))
                i += len(rows)

            samples = np.zeros((len(rows),
                                self.lookback // step,
                                data.shape[-1]))
            targets = np.zeros((len(rows), self.dimension))
            for j, row in enumerate(rows):
                indices = range(rows[j] - self.lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = label[rows[j] + self.delay]
            yield samples, targets

    def image_to_numpy(self, path):  # 把图片转成二值图后再转成list
        '''
        Nilm_dataprocess类的generator_image方法中调用的方法，用于加载已生成的vi轨迹图，外部无需单独调用此函数

        parameter
        ----------
        path(str):
            绝对路径+文件名

        return(list):
        ----------
            打开的图片转化为了list
        '''
        img = Image.open(path)
        # Img = PIL.ImageOps.invert(img)
        img = img.convert('1')
        image = img_to_array(img)
        return image

    def vicurve_to_numpy(self, volt_seq, current_seq, size_output):  # 把图片转成二值图后再转成list
        '''
        GUI采集到的v，i数据绘制成轨迹图，同时转换为numpy数组

        parameter
        ----------
        path(str):
            绝对路径+文件名

        return(list):
        ----------
            打开的图片转化为了list
        '''
        volt_seq = self.standlize_n1to1(volt_seq)
        current_seq = current_seq - np.mean(current_seq)
        current_seq = current_seq / 65.0
        # 使用plt进行画图

        # img = Image.open('00.jpg') #读取图片像素为512X512

        # fig=plt.figure("Image",frameon=False)  # 图像窗口名称

        # plt.imshow(img)
        fig = plt.figure(figsize=(size_output, size_output))
        plt.plot(volt_seq, current_seq, color='black')  # wave_data[0]是电压

        canvas = fig.canvas

        # 去掉图片四周的空白

        plt.axis('off')  # 关掉坐标轴为 off

        # 设置画布大小（单位为英寸），每1英寸有100个像素

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        # plt.show()

        buffer_ = io.BytesIO()
        fig.savefig(buffer_, format="png")
        buffer_.seek(0)
        img = Image.open(buffer_)
        img = img.convert('1')
        img = img_to_array(img)
        # 释放缓存
        buffer_.close()
        return img

    def generator_image(self, label, min_index, max_index, step=1):
        if max_index is None:
            pass
            # max_index = len(data) - self.delay - 1
        i = min_index + self.lookback
        while 1:
            csvfile = pd.read_excel(self.path_csv)
            if self.shuffle:
                rows = np.random.randint(min_index + self.lookback, max_index, size=self.batch_size)
            else:
                if i + self.batch_size >= max_index:
                    i = min_index + self.lookback
                rows = np.arange(i, min(i + self.batch_size, max_index))
                i += len(rows)
            samples_image = np.zeros((len(rows),
                                      self.lookback // step,
                                      32,
                                      32,
                                      1))
            targets = np.zeros((len(rows), self.dimension))
            # data_image_temp1=[]
            data_image_temp2 = []
            for j, row in enumerate(rows):
                data_image_temp2 = []
                indices = range(rows[j] - self.lookback, rows[j], step)  # row[j]对应csv的行数
                # count=0
                for lines in indices:
                    imagefilename = csvfile.iloc[lines, 0]
                    img = self.image_to_numpy(self.path_img + imagefilename)
                    # data_image_temp1.append(img)
                    # data_image_temp2.append(data_image_temp1)
                    data_image_temp2.append(img)
                    # data_image_temp1=[]
                    # count+=1
                # c=np.array(data_image_temp2).shape
                # samples_image[j,:,:,:,:]=data_image_temp2[:,:,:,:]
                samples_image[j] = data_image_temp2
                # samples_image[j].append(data_image_temp2)
                targets[j] = label[rows[j] + self.delay]
            yield samples_image, targets