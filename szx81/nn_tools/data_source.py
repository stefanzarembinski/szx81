import random
import math
import numpy as np

import core as co
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
        features = self.sinus_data(begin, count)
        target = features
        return features, target, indexes

        return (
            self.features[begin: end_index], 
            self.targets[begin: end_index],
            self.indexes[begin: end_index])


class ForexDataSource:
    def __init__(self, data, scalers):

        hd_values = list(data)
        self.scalers = scalers
        self.feature_count = 2

        self.indexes = []
        
        opens = []
        volumes = []
        for val in hd_values:
            self.indexes.append(int(val[0] // co.config.PERIOD))
            opens.append((val[1][0][0] + val[1][1][0]) / 2)
            volumes.append(val[2])

        (opens, volumes) = self.fit_transform(
            [np.array(opens).reshape(-1, 1), np.array(volumes).reshape(-1, 1)]
            )
        self.targets = opens
        self.features = np.concatenate((opens, volumes), axis=1)

    def fit_transform(self, data):
        for i in range(len(data)):
            data[i] = self.scalers[i].fit_transform(data[i])
        return data
    
    def len(self):
        return len(self.indexes)
    
    def get_data(self, end_index, count):
        begin = end_index - count - 1
        return (
            self.features[begin: end_index], 
            self.targets[begin: end_index],
            self.indexes[begin: end_index])
