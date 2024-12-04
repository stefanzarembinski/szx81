import random
import math

import hist_data as hd

class SinusDataSource:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(SinusDataSource, cls).__new__(cls)
        return cls._instance    

    def __init__(self, noise=0.0):
        self.noise = noise

    def len(self):
        return -1
    
    def sinus_data(self, begin, count):
        return [math.sin(i * .1) + random.uniform(
            -self.noise, self.noise) for i in range(begin, begin + count + 1)]

    def get_data(self, end_index, count, debug=False):
        begin = end_index - count - 1
        end = end_index 
        indexes = [i for i in range(begin, end)]      
        if debug:
           return indexes, indexes
        return self.sinus_data(begin, count), indexes

class ForexDataSource:
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ForexDataSource, cls).__new__(cls)
        return cls._instance  
    
    def __init__(self):
        hd.set_hist_data(data_count=None)
        hd_values = list(hd.DICT_DATA.values())
        self.data_x = []
        self.data_y = []
        for val in hd_values:
            y = (val[1][0][0] + val[1][1][0]) / 2
            self.data_y.append(y)
            # self.data_x.append((y, hd_values[i][2]))
            self.data_x.append(y)        

    def len(self):
        return len(self.data_x)
    
    def get_data(self, end_index, count, debug=False):
        begin = end_index - count - 1
        end = end_index 
        indexes = [i for i in range(begin, end)]      
        if debug:
           return indexes, indexes
        return self.data_x[begin: end], indexes

class ForexDiffDataSource:
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ForexDiffDataSource, cls).__new__(cls)
        return cls._instance  
    
    def __init__(self, future_count=10):
        hd.set_hist_data(data_count=None)
        hd_values = list(hd.DICT_DATA.values())
        self.data_x = []
        self.data_y = []
        for i in range(future_count, len(hd_values)):
            f = i - future_count
            x = (hd_values[f][1][0][0] + hd_values[f][1][1][0]) / 2
            x_f = (hd_values[i][1][0][0] + hd_values[i][1][1][0]) / 2
            self.data_y.append(x_f - x)
            # self.data_x.append((x, hd_values[i][2]))
            self.data_x.append(x)        

    def len(self):
        return len(self.data_x)
    
    def get_data(self, end_index, count, debug=False):
        begin = end_index - count - 1
        end = end_index 
        indexes = [i for i in range(begin, end)]      
        if debug:
           return indexes, indexes
        return self.data_x[begin: end], indexes

