import numpy as np
import random
import math
import time
import pickle
import os
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

class BatteryDataPreprocessor:
    count = 0
    def __init__(self, dir_path,battery_list, target):
        self.dir_path = dir_path
        self.battery_list = battery_list
        self.target = target 
        self.battery_data = {}
    ###########################################################################################################################################################################################
    ################################################################ NASA #####################################################################################################################
    
    def load_nasa_datasets(self):
        print("Loading NASA datasets...")
        for name in self.battery_list:
            print(f'Load Dataset {name}.mat ...')
            path = self.dir_path + name + '.mat'
            data = self.loadMat(path)
            self.battery_data[name] = self.getBatteryCapacity(data)
        print("Datasets loaded successfully.")

    # convert str to datatime 
    def convert_to_time(self, hmm):
        year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
        return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


    # load .mat data
    def loadMat(self, matfile):
        data = scipy.io.loadmat(matfile)
        filename = matfile.split("/")[-1].split(".")[0]
        col = data[filename]
        col = col[0][0][0][0]
        size = col.shape[0]

        data = []
        for i in range(size):
            k = list(col[i][3][0].dtype.fields.keys())
            d1, d2 = {}, {}
            if str(col[i][0][0]) != 'impedance':
                for j in range(len(k)):
                    t = col[i][3][0][0][j][0]
                    l = [t[m] for m in range(len(t))]
                    d2[k[j]] = l
            d1['type'], d1['ambient_temperature'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(self.convert_to_time(col[i][2][0])), d2
            data.append(d1)

        return data


    # get capacity data
    def getBatteryCapacity(self, Battery):
        cycle, capacity = [], []
        i = 1
        for Bat in Battery:
            if Bat['type'] == 'discharge':
                capacity.append(Bat['data']['Capacity'][0])
                cycle.append(i)
                i += 1
        return [cycle, capacity]


    # get the charge data of a battery
    def getBatteryValues(self, Battery, Type='charge'):
        data=[]
        for Bat in Battery:
            if Bat['type'] == Type:
                data.append(Bat['data'])
        return data

###########################################################################################################################################################################################
################################################################ PLOT #####################################################################################################################
    def plot_capacity_degradation(self, title='Capacity degradation at ambient temperature of 24°C'):
        fig, ax = plt.subplots(1, figsize=(12, 8))
        color_list = ['b:', 'g--', 'r-.', 'c.']

        if self.dataset_type == 'NASA':
            # Assuming NASA dataset 'Battery' structure: {name: [cycles, capacities]}
            for name, color in zip(self.battery_list, color_list):
                cycles, capacities = self.battery_data[name]
                ax.plot(cycles, capacities, color, label=name)     
        ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title=title)
        plt.legend()
        plt.show()
#TARGET = self.battery_list

def drop_outlier(array,count,bins):
    index = []
    range_ = np.arange(1,count,bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)


def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
def get_train_test(data_dict, name, window_size=8):
    data_sequence=data_dict[name]['capacity']
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v['capacity'], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re)/true_re if abs(true_re - pred_re)/true_re<=1 else 1


def evaluation(y_test, y_predict):
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return rmse
    
    
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_dataloader_NASA(data_path, batch_size=32):
    train_set, valid_set, test_set, = pickle.load(open(data_path), 'rb')
    train_data = BatteryDataPreprocessor(data_path, train_set, dataset_type='NASA')
    valid_data = BatteryDataPreprocessor(data_path, valid_set, dataset_type='NASA')
    test_data = BatteryDataPreprocessor(data_path, test_set, dataset_type='NASA')
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=1)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=1)

    return train_loader, valid_loader, test_loader