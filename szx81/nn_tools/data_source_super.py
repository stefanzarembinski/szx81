import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

import core as co
import hist_data as hd
hd.set_hist_data(data_count=None)

class DataSource:
    def __init__(self, data, scalers, **kwargs):
        
        self.data = list(data)
        self.scalers = scalers
        self.feature_count = 2
        self.end_index = None
        self.data_count = None
        self.features = None
        self.targets = None
        self.indexes = None
 
        self.fit_data(**kwargs)

    def fit_data(self):
        return

    def feature_names(self):
        return [str(i) for i in range(20)]
        # Change accordingly
    
    def target_names(self):
        return [str(i) for i in range(20)]
        # Change accordingly   

    def fit(self, data):
        for i in range(len(data)):
            if data[i] is not None:
                data[i] = self.scalers[i].fit(data[i])
        return data

    def fit_transform(self, data):
        for i in range(len(data)):
            if data[i] is not None:
                data[i] = self.scalers[i].fit_transform(data[i])
        return data
    
    def transform(self, data, index=None):
        if index is None:
            for i in range(len(data)):
                if data[i] is not None:
                    data[i] = self.scalers[i].transform(data[i])
        else:
            data = self.scalers[index].transform(data)
        return data
    
    def inverse_transform(self, data, index=None):
        if index is None:
            for i in range(len(data)):
                if data[i] is not None:
                    data[i] = self.scalers[i].inverse_x(data[i])
        else:
            data = self.scalers[index].inverse_x(data)        
        return data
    
    def len(self):
        return len(self.indexes)
    
    def get_data__(self):
        # Change accordingly   
        return (
            self.features, 
            self.targets, 
            self.indexes)
    
    def get_data(self, end_index, data_count, future_count):
        """
        Returns amount of data, typically called by a ``Context Sequencer`` 
        or ``Predictor``.

        Parameters:
        -----------
        end_index : Index of oldest historical data used, including targeting.
        data_count : Index of the earliest historical data used is 
            `end_index - data_count` (possibly lowered in order to meet 
            constrictions.)
        future_count : Historical data index difference between training
            and prediction data

        Returns:
        --------

        (features, targets, indexes) : Tuple of numpy arrays containing
            data relevant (yet not formatted) for nn modeling.
        """
        self.end_index = end_index
        self.data_count = data_count
        self.future_count = future_count

        feature_data_count = self.data_count - self.feature_count

        if feature_data_count % self.step != 0:
            feature_data_count = (feature_data_count // self.step + 1) \
                * self.step
        self.begin = self.end_index - feature_data_count - future_count
        if self.begin + 1 < 0 or self.end_index + 1 > len(self.data):
            raise Exception(f''' 
ERROR
The wanted data range is 
({self.begin + 1}, {self.end_index + 1})
while the maximal is
({0}, {len(self.data)})
''')

        return self.get_data__()
        
    def report(self, verbose=False):
        index_count = self.indexes[-1] - self.indexes[0]
        print(f'''
wanted end index: ({self.end_index})
wanted data count {self.data_count}
index range: ({self.indexes[0]}, {self.indexes[-1]})
index count: {self.indexes[-1] - self.indexes[0]}''')
        if index_count != self.data_count:
            print('Real data count is adjusted to constrictions.')
        else:
            print()
        if verbose:
            print(f'''
features:
{self.features[:3]}
targets:
{self.targets[:3]}
indexes:
{self.indexes[:3]}
''')

    def plot(self):
        targets = self.targets.transpose()
        for i in range(len(targets)):
            plt.plot(self.indexes, targets[i], label=self.target_names()[i])
        features = self.features.transpose()
        for i in range(len(features)):
            plt.plot(self.indexes, features[i], label=self.feature_names()[i])
        plt.legend()
        plt.show()
