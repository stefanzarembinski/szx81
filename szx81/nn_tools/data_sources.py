import random
import math
import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# https://realpython.com/python-logging/
# https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
# logging.basicConfig(format="{levelname}:{funcName}: {message}", style="{")

import logging
log = logging.getLogger(__file__)
log.setLevel(level=logging.DEBUG)


import core as co
import hist_data as hd

from nn_tools.data_source_super import DataSource

def set_logging_level(level=logging.DEBUG):
    logger = logging.getLogger(__file__)
    logger.setLevel(level)

class OpenVolumeDs(DataSource):
    def fit_data__(self):
        self.step = self.kwargs['step']
        feature_0 = []
        feature_1 = []

        for i in range(0, len(self.data)):  
            open = 0
            volume = 0
            if not i + self.step < len(self.data):
                break
            for k in range(self.step):
                val = self.data[i + k]
                open += (val[1][0][0] + val[1][1][0]) / 2
                volume += val[2]              

            feature_0.append(open / self.step)
            feature_1.append(volume / self.step)

        feature_0 = np.array(feature_0)
        feature_1 = np.log(np.array(feature_1) + 10.0)

        self.fit_transform([feature_0, feature_1])

    def feature_names(self):
        return ('feature_0', 'feature_1')
    
    def target_names(self):
        return ('target_0',)
        
    def get_data__(self):
        def helper(i):
            val = self.data[i]
            return (val[1][0][0] + val[1][1][0]) / 2, val[2]

        indexes_tf = [] 
        feature_0 = []
        feature_1 = []
        targets = []
        # for i in range(len(self.indexes)):
        #     index = 0
        #     open = 0
        #     volume = 0
        #     target = 0

        #     for k in range(self.step):
        #         if k == self.step - 1:
        #             _open, _volume = helper(i + k)
        #             open += _open
        #             volume += _volume
        #             _open, _volume = helper(i + k + self.future_count)
        #             target += _open
        #             index = i + k                    
        #             indexes_tf.append(index)
        #             feature_0.append(open / self.step)
        #             feature_1.append(volume / self.step)
        #             targets.append(target / self.step)


        for i in range(self.begin_index, self.end_index, self.step):
            index = 0
            open = 0
            volume = 0
            target = 0
            
            for k in range(self.step):
                if k == self.step - 1:
                    _open, _volume = helper(i + k)
                    open += _open
                    volume += _volume
                    _open, _volume = helper(i + k + self.future_count)
                    target += _open
                    index = i + k                    
                    indexes_tf.append(index)
                    feature_0.append(open / self.step)
                    feature_1.append(volume / self.step)
                    targets.append(target / self.step)

        feature_0 = np.array(feature_0)
        feature_1 = self.log_volume(np.array(feature_1))
        feature_0, feature_1 = self.transform([feature_0, feature_1])

        targets = np.array(targets)
        targets = self.transform(targets, index=0)

        self.features = np.concatenate((feature_0, feature_1), axis=1)
        self.targets = targets
        self.indexes_tf = np.array(indexes_tf)
        return (
            self.features, 
            self.targets, 
            self.indexes_tf)

class OpenDs(DataSource):

    def fit_data__(self):
        self.step = self.kwargs['step']
        self.scaler_opens = self.scalers[0]
        self.scaler_volumes = None
        feature_0 = []
        for i in range(0, len(self.data)):
            val = self.data[i]
            feature_0.append((val[1][0][0] + val[1][1][0]) / 2)
            open = 0
            if not i + self.step < len(self.data): 
                break
            for k in range(self.step):
                val = self.data[i + k]
                open += (val[1][0][0] + val[1][1][0]) / 2           

            feature_0.append(open / self.step)

        feature_0 = np.array(feature_0)
        feature_0 = np.array(feature_0)
        log.debug(f'''
before transformation
len(self.data): {len(self.data)}
len(feature_0): {len(feature_0)}
feature_0(min. max) = ({np.min(feature_0):.1e}, {np.max(feature_0):.1e})
''')

        feature_0 = self.fit_transform(feature_0, 0)
        feature_0 = self.transform(feature_0, 0)

        log.debug(f'''
after transformation
len(feature_0): {len(feature_0)}                  
feature_0(min, max): ({np.min(feature_0):.1e}, {np.max(feature_0):.1e})''')


    def feature_names(self):
        return ('open',)
    
    def target_names(self):
        return ('open',)
    
    def set_data_begin(self):
        self.begin_index = self.end_index \
            - (self.data_count + (self.future_count + self.step))

    def get_data__(self):
        indexes_tf = []
        target_0 = []
        feature_0 = []

        def helper(j):
            val = self.data[j]
            return (val[1][0][0] + val[1][1][0]) / 2
        
        for i in range(self.begin_index, self.end_index):
            _feature_0 = 0
            _target_0 = 0
            for k in range(self.step):
                _feature_0 += helper(i + k)
                _target_0 += helper(i + k + self.future_count)    
                if k == self.step - 1:
                    indexes_tf.append(i + k)
                    feature_0.append(_feature_0 / self.step)
                    target_0.append(_target_0 / self.step)

        indexes_tf = np.array(indexes_tf)
        feature_0 = np.array(feature_0)

        feature_0 = self.transform(feature_0, 0)
        target_0 = np.array(target_0)
        target_0 = self.transform(target_0, 0)

        log.debug(f'''
len(feature indexes): {len(indexes_tf)}
feature indexes(min. max): ({np.min(indexes_tf)}, {np.max(indexes_tf)})
len(feature_0): {len(feature_0)}
feature_0(min. max): ({np.min(feature_0):.1e}, {np.max(feature_0):.1e})
len(target_0): {len(target_0)}
target_0(min. max): ({np.min(target_0):.1e}, {np.max(target_0):.1e})
''')
        self.features = np.concatenate((feature_0,), axis=1)
        self.targets = np.concatenate((target_0,), axis=1)
        self.indexes_tf = indexes_tf

        opens = None
        if self.opens is not None:
            opens = self.opens[self.begin_index: self.end_index]
        volumes = None
        if self.volumes is not None:
            volumes = self.volumes[self.begin_index: self.end_index]
        dt = DataSource.DataTransfer(
            data=self.data.copy(),
            features=self.features,
            targets=self.targets,
            indexes_tf=self.indexes_tf,
            opens=opens,
            volumes=volumes,
            data_range=(self.begin_index, self.end_index)
        )
        return dt

class SinusDs(DataSource):

    class Sinus:
        def __init__(self, noise=0.03, stop=10000, start=0, step=1):
            self.noise = noise
            self.start = start
            self.step = step
            self.stop = stop 

        def __getitem__(self, i):
            # import pdb; pdb.set_trace()
            if isinstance(i, slice):
                if i.start is not None:
                    self.start = i.start
                if i.step is not None:
                    raise ValueError(
                        f'slice step is {i.step}; it can be 1, only ')
                if i.stop is not None:
                    self.stop = i.stop

                return SinusDs.Sinus(
                    self.noise, self.stop, self.start, self.step)
            if i < 0:
                i = self.stop + i
            
            k = (i + self.start)
            if k < self.start or k >= self.stop:
                raise IndexError(f'index is {k} while it has to be within limits [{self.start}, {self.stop}[')
            return .5 * math.sin(k * .03 * self.step) \
                    + random.uniform(-self.noise, self.noise) + .5
        
        def __len__(self):
            return self.stop - self.start
        
        def __str__(self):
            return f'''
start index: {self.start}
step: {self.step}
stop index: {self.stop}
length: {len(self)}
'''

    def fit_data__(self):
        self.step = self.kwargs['step']
        self.scaler_opens = None
        self.scaler_volumes = None

    def feature_names(self):
        return ('sinus features',)
    
    def target_names(self):
        return ('sinus targets',)
    
    def get_data__(self):
    
        indexes_tf = []
        target_0 = []
        feature_0 = []

        end = self.end_index - 1
        future_count = 0 if self.is_testing else self.future_count
        if self.is_testing:
            future_count = 0
        else:
            future_count = self.future_count

        for i in range(self.data_count):
            _feature_0 = 0
            _target_0 = 0
            for k in range(self.step):
                _feature_0 += self.data[end - (i + k + future_count)]
                _target_0 += self.data[end - (i + k)]
                if k == 0: indexes_tf.insert(0, end - (i + k))
                if k == self.step - 1:
                    feature_0.insert(0, (_feature_0 / self.step))
                    target_0.insert(0, (_target_0 / self.step)) 

        self.begin = indexes_tf[0] - self.step

        feature_0 = np.array(feature_0).reshape(-1, 1)
        target_0 = np.array(target_0).reshape(-1, 1)

        self.features = np.concatenate((feature_0,), axis=1)
        self.targets = np.concatenate((target_0,), axis=1)
        self.indexes_tf = indexes_tf

        opens = None
        volumes = None
        data=self.data
        if hasattr(self.data, 'copy'):
            data=data.copy()
        
        dt = DataSource.DataTransfer(
            data=data,
            features=self.features,
            targets=self.targets,
            indexes_tf=self.indexes_tf,
            opens=opens,
            volumes=volumes,
            data_range=(self.begin_index, self.end_index)
        )
        return dt
    
def test_ds():
    # ds = OpenVolumeDs(
    #     hd.DICT_DATA.values(), (StandardScaler(), StandardScaler()), step=10)
    # ds = OpenDs(
    #     hd.DICT_DATA.values(), (MinMaxScaler(), MinMaxScaler()), step=3) 

    ds = SinusDs(SinusDs.Sinus(noise=0.03, len=5000), (None, None), step=5)    
    ds.get_data(end_index=1000, data_count=30, future_count=3)
    ds.report()
    # ds.plot_ds(plt, data_count=10, show=True)
    ds.plot(data_count=50)


def main():
    test_ds()

if __name__ == "__main__":
    main()  