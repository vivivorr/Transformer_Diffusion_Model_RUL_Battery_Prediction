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
################################################################ CALCE ####################################################################################################################
    def load_calce_datasets(self):
        print("Loading datasets ...")
        for name in self.battery_list:
            print('Load Dataset ' + name + ' ...')
            
            discharge_capacities, health_indicator, internal_resistance, CCCT, CVCT = [], [], [], [], []
            path = glob.glob(self.dir_path + name + '/*.xlsx')
            dates = [pd.read_excel(p, sheet_name=1)['Date_Time'][0] for p in path]
            idx = np.argsort(dates)
            path_sorted = np.array(path)[idx]

            for p in path_sorted:
                df = pd.read_excel(p, sheet_name=1)
                self.process_battery_data(df,discharge_capacities, health_indicator, internal_resistance, CCCT, CVCT)
            
            self.battery_data[name] = self.aggregate_data(discharge_capacities, health_indicator, internal_resistance, CCCT, CVCT)

        print("Datasets loaded successfully.")

    def process_battery_data(self, df, discharge_capacities, health_indicator, internal_resistance, CCCT, CVCT):
        cycles = list(set(df['Cycle_Index']))
        for c in cycles:
            df_lim = df[df['Cycle_Index'] == c]
            df_cc = df_lim[df_lim['Step_Index'] == 2] # Constant current
            df_cv = df_lim[df_lim['Step_Index'] == 4] # Constant Voltage
            CCCT.append(np.max(df_cc['Test_Time(s)'])-np.min(df_cc['Test_Time(s)'])) # Time spent in CC
            CVCT.append(np.max(df_cv['Test_Time(s)'])-np.min(df_cv['Test_Time(s)'])) # Time spent in CV
            
            # Discharging
            df_d = df_lim[df_lim['Step_Index'] == 7]
            if not df_d.empty:
                d_v = df_d['Voltage(V)'].to_numpy()
                d_c = df_d['Current(A)'].to_numpy()
                d_t = df_d['Test_Time(s)'].to_numpy()
                # Calculate discharge capacity
                time_diff = np.diff(d_t)
                discharge_capacity = np.cumsum(time_diff * d_c[1:] / 3600)  # Convert to Ah
                discharge_capacities.append(-discharge_capacity[-1])
                start_capacity = discharge_capacity[np.abs(d_v[1:] - 3.8).argmin()]
                end_capacity = discharge_capacity[np.abs(d_v[1:] - 3.4).argmin()]
                health_indicator.append(-start_capacity + end_capacity)
                d_im = df_d['Internal_Resistance(Ohm)'].to_numpy()
                internal_resistance.append(np.mean(d_im))
                BatteryDataPreprocessor.count += 1

    def aggregate_data(self, discharge_capacities, health_indicator, internal_resistance, CCCT, CVCT):
        discharge_capacities = np.array(discharge_capacities)
        health_indicator = np.array(health_indicator)
        internal_resistance = np.array(internal_resistance)
        CCCT = np.array(CCCT)
        CVCT = np.array(CVCT)
    
        idx = drop_outlier(discharge_capacities, BatteryDataPreprocessor.count, 40)
        data = pd.DataFrame({'cycle':np.linspace(1,idx.shape[0],idx.shape[0]),
                              'capacity':discharge_capacities[idx],
                              'SoH':health_indicator[idx],
                              'resistance':internal_resistance[idx],
                              'CCCT':CCCT[idx],
                              'CVCT':CVCT[idx]})
        df = pd.DataFrame(data)
        return df
    def drop_outlier(self, array,count,bins):
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

###########################################################################################################################################################################################
################################################################ PLOT #####################################################################################################################
    def plot_capacity_degradation(self, title='Capacity degradation at ambient temperature of 24°C'):
        fig, ax = plt.subplots(1, figsize=(12, 8))
        color_list = ['b:', 'g--', 'r-.', 'c.']

        for name, color in zip(self.battery_list, color_list):
            df_result = self.battery_data[name]
            # Update this line if CALCE dataset structure is different
            ax.plot(df_result['cycle'], df_result['capacity'], color, label=f'Battery_{name}')
        
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




def get_dataloader_CALCE(data_path, batch_size=32):
    train_set, valid_set, test_set, = pickle.load(open(data_path), 'rb')
    train_data = BatteryDataPreprocessor(data_path, train_set)
    valid_data = BatteryDataPreprocessor(data_path, valid_set)
    test_data = BatteryDataPreprocessor(data_path, test_set)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=1)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=1)

    return train_loader, valid_loader, test_loader