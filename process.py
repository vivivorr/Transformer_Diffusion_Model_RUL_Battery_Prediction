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
    def __init__(self, dir_path,battery_list, dataset_type, target):
        self.dir_path = dir_path
        self.battery_list = battery_list
        self.dataset_type = dataset_type
        self.target = target 
        self.battery_data = {}

    def load_datasets(self):
        if self.dataset_type == 'CALCE':
            self.load_calce_datasets()
        elif self.dataset_type == 'NASA':
            self.load_nasa_datasets()
        else:
            raise ValueError("Unsupported dataset type.")

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
############################################################ TRIPLET GENERATION ###########################################################################################################    
    # def generate_triplets(self, input_size=100, feature='Capacity'):
    #     all_triplets = []
    #     if self.dataset_type == 'CALCE':
    #         for name in self.battery_list:
    #             df_result = self.battery_data[name]
    #             cycles = df_result['cycle'].to_numpy()
    #             capacities = df_result['capacity'].to_numpy()
    #             mask_bits = np.ones(len(cycles))
    #             triplets = [(feature, cycle, capacity, mask) for cycle, capacity, mask in zip(cycles, capacities, mask_bits)]
    #             if len(triplets) > input_size:
    #                 selected_indices = np.random.choice(len(triplets), size=input_size, replace=False)
    #                 triplets = [triplets[i] for i in selected_indices]
    #             all_triplets.extend(triplets)
    #     elif self.dataset_type == 'NASA':
    #         for name in self.battery_list:
    #             cycles, capacities = self.battery_data[name]
    #             triplets = [(feature, cycle, capacity, 1) for cycle, capacity in zip(cycles, capacities)]
    #             if len(triplets) > input_size:
    #                 selected_indices = np.random.choice(len(triplets), size=input_size, replace=False)
    #                 triplets = [triplets[i] for i in selected_indices]
    #             all_triplets.extend(triplets)
    #         return all_triplets
        
    # def generate_triplets_v2(self, data, target_feature, max_triplets):
    #     triplets_data = []
    #     target_data = []
    #     correlation = data.corr()[target_feature].abs().sort_values(ascending=False)
    #     correlation.drop(target_feature, inplace=True)  # Avoid self-correlation
        
    #     for index, row in data.iterrows():
    #         cycle = row['cycle']
    #         for feature in data.columns.difference(['cycle']):
    #             if feature == target_feature:
    #                 target_data.append((cycle, row[target_feature] if pd.notnull(row[target_feature]) else 0, 1 if pd.notnull(row[target_feature]) else 0))
    #             else:
    #                 value = row[feature]
    #                 mask = 1 if pd.notnull(value) else 0  # Set mask to 1 if data is present, 0 otherwise
    #                 triplets_data.append((feature, cycle, value if mask else 0, mask))

    #     if len(triplets_data) > max_triplets:
    #         max_features = max_triplets // len(data['cycle'].unique())
    #         selected_features = correlation.index[:max_features].tolist()
    #         triplets_data = [t for t in triplets_data if t[0] in selected_features]

    #     while len(triplets_data) < max_triplets:
    #         random_index = np.random.choice(data.index)
    #         cycle = data.loc[random_index, 'cycle']
    #         available_features = data.columns[data.loc[random_index].notnull()].difference(['cycle', target_feature])
    #         feature = np.random.choice(available_features)
    #         value = data.loc[random_index, feature]
    #         mask = 1 
    #         triplets_data.append((feature, cycle, value, mask))
        
    #     dtypes_triplets = np.dtype([
    #         ('Feature', 'U50'),  
    #         ('Cycle', np.int_),   
    #         ('Value', np.float_), 
    #         ('Mask', np.int_)     
    #     ])
    #     dtypes_target = np.dtype([
    #         ('Cycle', np.int_),   
    #         ('Value', np.float_), 
    #         ('Mask', np.int_)   
    #     ])
        
    #     triplets_x_array = np.array(triplets_data, dtype=dtypes_triplets)
    #     target_values_array = np.array(target_data, dtype=dtypes_target)

    #     return triplets_x_array, target_values_array

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
        else:
            # Assuming CALCE dataset structure or similar where data is in DataFrame
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


# def get_dataloader(data_path, var_path, size, batch_size=32):
#     train_set, train_info, valid_set, valid_info, test_set, test_info = pickle.load(open(data_path, 'rb'))
#     var, target_var = pickle.load(open(var_path, 'rb'))
#     train_data = MIMIC_Dataset(train_set, train_info, size, target_var)
#     valid_data = MIMIC_Dataset(valid_set, valid_info, size, target_var)
#     test_data = MIMIC_Dataset(test_set, test_info, size, target_var)
    
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=1)
#     valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=1)
#     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=1)
    
#     return train_loader, valid_loader, test_loader

def get_dataloader_CALCE(data_path, batch_size=32):
    train_set, valid_set, test_set, = pickle.load(open(data_path), 'rb')
    train_data = BatteryDataPreprocessor(data_path, train_set)
    valid_data = BatteryDataPreprocessor(data_path, valid_set)
    test_data = BatteryDataPreprocessor(data_path, test_set)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=1)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=1)

    return train_loader, valid_loader, test_loader

def get_dataloader_NASA(data_path, batch_size=32):
    train_set, valid_set, test_set, = pickle.load(open(data_path), 'rb')
    train_data = BatteryDataPreprocessor(data_path, train_set, dataset_type='NASA')
    valid_data = BatteryDataPreprocessor(data_path, valid_set, dataset_type='NASA')
    test_data = BatteryDataPreprocessor(data_path, test_set, dataset_type='NASA')
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=1)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=1)

    return train_loader, valid_loader, test_loader