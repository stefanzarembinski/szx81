import random
import math
import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# https://realpython.com/python-logging/
# https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
import logging
# logging.basicConfig(format="{levelname}:{funcName}: {message}", style="{")
log = logging.getLogger(__file__)
log.setLevel(level=logging.DEBUG)

# log = logging.getLogger(__file__)
# log.setLevel(level=logging.DEBUG)

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
    
    def raw_data_begin(self):
        self.feature_data_count = self.data_count + self.step
        return self.end_index \
            - (self.feature_data_count + (self.future_count + self.step))
        
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
    
    def plot_ds(self, plt, data_count=10, show=False):
        """
        Parameters:
        -----------
        predictions : Array-like object containing numpy arrays.
        """
        data_count = min(len(self.indexes_tf), data_count)
        indexes_tf = self.indexes_tf[-data_count:]
        targets = self.targets.transpose()
        for i in range(len(targets)):
            plt.plot(
                indexes_tf + self.feature_count, 
                targets[i][-data_count:], linewidth=7, alpha=0.5, # marker='o', 
                     label=self.target_names()[i])
        features = self.features.transpose()
        for i in range(len(features)):
            plt.plot(
                indexes_tf, 
                features[i][-data_count:], # marker='x', 
                     label=self.feature_names()[i])
        plt.vlines(
            indexes_tf[-1], np.min(targets[0]), np.max(targets[0]), 
            linestyle='dashed',
            label=f'time now, future count is {self.feature_count}')
        
        if show:
            plt.legend()
            plt.show()
        
        return self.indexes_tf[-data_count:][0]

class SinusDs(DataSource):

    def fit_data__(self):
        self.step = self.kwargs['step']
        self.noise_strength = self.kwargs['noise']
        self.scaler_opens = None
        self.scaler_volumes = None

    def feature_names(self):
        return ('sinus features',)
    
    def target_names(self):
        return ('sinus targets',)
    
    def raw_data_begin(self):
        self.feature_data_count = self.data_count + self.step
        return self.end_index \
            - (self.feature_data_count + (self.future_count + self.step))
    
    def get_data__(self):
        indexes_tf = []
        target_0 = []
        feature_0 = []
        
        for i in range(self.begin_index, self.end_index):
            _feature_0 = 0
            _target_0 = 0
            for k in range(self.step):
                _feature_0 += self.data(i + k, self.noise_strength)
                _target_0 += self.data(
                    i + k + self.future_count, self.noise_strength)    
                if k == self.step - 1:
                    indexes_tf.append(i + k)
                    feature_0.append(_feature_0 / self.step)
                    target_0.append(_target_0 / self.step)

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
    
    def plot_ds(self, plt, data_count=10, show=False):
        """
        Parameters:
        -----------
        predictions : Array-like object containing numpy arrays.
        """
        data_count = min(len(self.indexes_tf), data_count)
        indexes_tf = np.array(self.indexes_tf[-data_count:])
        targets = self.targets.transpose()
        for i in range(len(targets)):
            plt.plot(
                indexes_tf + self.feature_count, 
                targets[i][-data_count:], linewidth=7, alpha=0.5, # marker='o', 
                     label=self.target_names()[i])
        features = self.features.transpose()
        for i in range(len(features)):
            plt.plot(
                indexes_tf, 
                features[i][-data_count:], # marker='x', 
                     label=self.feature_names()[i])
        plt.vlines(
            indexes_tf[-1], np.min(targets[0]), np.max(targets[0]), 
            linestyle='dashed',
            label=f'time now')
        log.debug(f'''
``future`` is {self.feature_count} long; 
``future_count`` is {self.feature_count}.

``time now`` (the last features index) is {indexes_tf[-1]}; 
``end_index`` is {self.end_index}.

''step`` is {self.step}
''')
        
        if show:
            plt.legend()
            plt.show()
        
        return self.indexes_tf[-data_count:][0]

def test_ds():
    # ds = OpenVolumeDs(
    #     hd.DICT_DATA.values(), (StandardScaler(), StandardScaler()), step=10)
    # ds = OpenDs(
    #     hd.DICT_DATA.values(), (MinMaxScaler(), MinMaxScaler()), step=3)
    
    sinus = lambda j, noise: .5 * math.sin(j * .03) + random.uniform(
        -noise, noise) + .5
    ds = SinusDs(
        sinus, (None, None), step=5, noise=0.01)    
    
    ds.get_data(end_index=1000, data_count=30, future_count=3)
    ds.report()
    # ds.plot_ds(plt, data_count=10, show=True)
    ds.plot(data_count=50)


def main():
    test_ds()

if __name__ == "__main__":
    main()  