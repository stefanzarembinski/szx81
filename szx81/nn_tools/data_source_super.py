import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

import core as co
import hist_data as hd
hd.set_hist_data(data_count=None)

class DataSource:
    def __init__(self, data, scalers, **kwargs):
        """ The ``get_data`` method returns the given count of data object 
        needed for composing feature and target inputs of an NN model.

        Parameters
        ----------
        data : Array of historical data available.
        scalers: Array of NN model inputs scalers.
        """
        
        self.data = list(data)
        self.scalers = scalers
        self.feature_count = 2
        self.end_index = None
        self.begin_index= None
        self.feature_data_count = None
        self.data_count = None
        self.features = None
        self.targets = None
        self.indexes = None
        self.opens_hist = None
        self.volumes_hist = None
        self.indexes_hist = None
        self.kwargs = kwargs
        self.scaler_opens = MinMaxScaler()
        self.scaler_volumes = MinMaxScaler()

        self.scaler_opens.fit(np.array(
            [(_[1][0][0] + _[1][1][0]) / 2 for _ in data]).reshape(-1, 1))
        self.scaler_volumes.fit(
            self.log_volume(np.array([_[2] for _ in data]).reshape(-1, 1)))

        self.fit_data()

    def fit_data(self):
        raise NotImplementedError()
    
    def feature_names(self):
        raise NotImplementedError()
        return [str(i) for i in range(20)]
    
    def raw_data_begin(self):
        """Calculates ``beginning index of historical data needed and returns 
        it.
        """
        raise NotImplementedError()
        return self.begin_index
    
    def get_data__(self):
        raise NotImplementedError()
        return (
            self.features, 
            self.targets, 
            self.indexes)

    def target_names(self):
        return [str(i) for i in range(20)]
        # Change accordingly   

    def plot_ds(self, plt):
        raise NotImplementedError()
    
    def log_volume(self, volume):
        return np.log(volume + 10)
    
    def fit(self, data):
        for i in range(len(data)):
            if data[i] is not None:
                data[i] = self.scalers[i].fit(data[i].reshape(-1, 1))
        return data

    def fit_transform(self, data):
        for i in range(len(data)):
            if data[i] is not None:
                data[i] = self.scalers[i].fit_transform(data[i].reshape(-1, 1))
        return data
    
    def transform(self, data, index=None):
        if index is None:
            for i in range(len(data)):
                if data[i] is not None:
                    data[i] = self.scalers[i].transform(data[i].reshape(-1, 1))
        else:
            data = self.scalers[index].transform(data.reshape(-1, 1))
        return data
    
    def inverse_transform(self, data, index=None):
        if index is None:
            for i in range(len(data)):
                if data[i] is not None:
                    data[i] = self.scalers[i].inverse_x(data[i].reshape(-1, 1))
        else:
            data = self.scalers[index].inverse_x(data.reshape(-1, 1))        
        return data
    
    def len(self):
        return len(self.indexes)
    
    def get_data(self, end_index, data_count, future_count):
        """
        Returns the given count of data object needed for composing feature and 
        target inputs of an NN model.

        Parameters:
        -----------
        end_index : Index of oldest historical data used, including targeting.
        data_count : Number of data units needed for imputing a NN model.
        future_count : Historical data index difference between training
            and prediction data

        Returns:
        --------

        (features, targets, indexes) : Tuple of numpy arrays containing
            data ordered.
        """
        self.end_index = end_index + 1
        self.data_count = data_count
        self.future_count = future_count
        self.begin_index= self.raw_data_begin()

        if self.begin_index+ 1 < 0 or self.end_index + 1 > len(self.data):
            raise Exception(f''' 
ERROR
The wanted data range is 
({self.begin_index+ 1}, {self.end_index + 1})
while the maximal is
({0}, {len(self.data)})
''')
        data = self.data[self.begin_index: self.end_index]
        self.opens_hist = np.array(
            [(_[1][0][0] + _[1][1][0]) / 2 for _ in data])
        self.volumes_hist = self.log_volume(np.array([_[2] for _ in data]))
        self.indexes_hist = np.array(
            [i for i in range(self.begin_index, self.end_index)])

        return self.get_data__()
        
    def report(self, verbose=False):
        index_count = self.indexes[-1] - self.indexes[0]
        print(f'''
REPORT: 
wanted end index: ({self.end_index})
wanted data count {self.data_count}
hist. data index range: ({self.indexes_hist[0]}, {self.indexes_hist[-1]})
features:
{self.features[:3]}
targets:
{self.targets[:3]}
indexes:
{self.indexes[:3]}
''')

    def plot(self, data_count=10):
        first_index = self.plot_ds(plt, data_count)
        # import pdb; pdb.set_trace()
        index = np.where(self.indexes_hist> first_index)[0][0] - 1
        opens_hist = self.scaler_opens.transform(self.opens_hist.reshape(-1, 1))
        plt.plot(
            self.indexes_hist[index:], 
            opens_hist[index:], label='open hist')
        volumes_hist = self.scaler_volumes.transform(
                                            self.volumes_hist.reshape(-1, 1))
        plt.plot(
            self.indexes_hist[index:], 
            volumes_hist[index:], label='log(volume hist.)')
        
        plt.legend()
        plt.show()
