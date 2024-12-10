import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

import core as co
import hist_data as hd
if hd.DATA is None:
    hd.set_hist_data(data_count=None)

class DataSource:
    class DataTransfer:
        """
        Data transfer object.

        Parameters:
        ----------
        indexes_tf : Index array for ``targets`` and ``features``.
        data : Historical data.
        data_range : Range if indexes for ``opens`` and ``volumes``.
        """
        def __init__(self,
                    data,
                    features,
                    targets,
                    indexes_tf,
                    opens,
                    volumes,
                    data_range):
            self.data = data
            self.features = features
            self.targets = targets
            self.indexes_tf = indexes_tf
            self.opens = opens
            self.volumes = volumes
            self.data_range = data_range

        def get_indexes(self):
            return np.array(
            [i for i in range(self.data_range[0], self.data_range[1])])

    def __init__(self, data, scalers, **kwargs):
        """ The ``get_data`` method returns the given count of data object 
        needed for composing feature and target inputs of an NN model.

        Parameters
        ----------
        data : Array of historical data available.
        scalers: Array of NN model inputs scalers.
        """
        self.data = data
        try:
            self.data = list(data)
        except:
            pass
        self.scalers = scalers
        self.feature_count = 2
        self.future_count = 10   
        self.kwargs = kwargs
        self.scaler_opens = MinMaxScaler()
        self.scaler_volumes = MinMaxScaler()
        self.end_index = None
        self.begin_index= None
        self.data_count = None
        self.features = None
        self.targets = None
        self.indexes_tf = None
        self.opens = None
        self.volumes = None
        self.indexes = None

        self.fit_data()           
    

    def fit_data__(self):
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
            self.indexes_tf)

    def target_names(self):
        return [str(i) for i in range(20)]
        # Change accordingly   

    def plot_ds(self, plt):
        raise NotImplementedError()
    
    def log_volume(self, volume):
        return np.log10(volume + 10)
    
    def fit(self, data, index=None):
        if index is None:
            for i in range(len(data)):
                if data[i] is not None:
                    data[i] = self.scalers[i].fit(data[i].reshape(-1, 1))
        else:
            data = self.scalers[index].fit(data.reshape(-1, 1))
        return data

    def fit_transform(self, data, index=None):
        if index is None:
            for i in range(len(data)):
                if data[i] is not None:
                    data[i] = self.scalers[i].fit_transform(
                        data[i].reshape(-1, 1))
        else:
            data = self.scalers[index].fit_transform(data.reshape(-1, 1))
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
        return len(self.indexes_tf)
    
    def fit_data(self):
      
        self.fit_data__()

        if (self.opens is None) and (self.scaler_opens is not None):
            self.opens = np.array(
                [(_[1][0][0] + _[1][1][0]) / 2 for _ in self.data]).reshape(
                    -1, 1)
            self.opens = self.scaler_opens.fit_transform(self.opens)
            
        if (self.volumes is None) and (self.scaler_volumes is not None):
            self.log_volume = self.log_volume(
                np.array([_[2] for _ in self.data]).reshape(-1, 1))
            self.volumes = self.scaler_volumes.fit()

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

        (features, targets, indexes_tf) : Tuple of numpy arrays containing
            data ordered.
        """
        self.end_index = end_index
        self.data_count = data_count
        self.future_count = future_count
        self.begin_index= self.raw_data_begin()

        if hasattr(self.data, '__len__'):
            if self.begin_index + 1 < 0 or self.end_index + 1 > len(self.data):
                raise Exception(f''' 
ERROR
The wanted data range is 
({self.begin_index+ 1}, {self.end_index + 1})
while the maximal is
({0}, {len(self.data)})
''')      
        return self.get_data__()
        
    def report(self, verbose=False):
        index_count = self.indexes_tf[-1] - self.indexes_tf[0]
        print(f'''
REPORT: 
wanted end index: ({self.end_index})
wanted data count {self.data_count}
hist. data index range (end excluded): ({self.begin_index}, {self.end_index})
features:
{self.features[:3]}
targets:
{self.targets[:3]}
feature indexes:
{self.indexes_tf[:3]}
''')

    def plot(self, data_count=10, future_margin=10):
        first_index = self.plot_ds(plt, data_count)
        # import pdb; pdb.set_trace()
        indexes = np.array(
            [i for i in range(self.begin_index, self.end_index \
                                                    + future_margin)])
        index = np.where(indexes > first_index)[0][0] - 1
        if self.opens is not None:
            opens = self.opens[self.begin_index: self.end_index \
                                                    + future_margin]
            plt.plot(indexes[index:], opens[index:], label='open orig.')
        if self.volumes is not None:
            volumes = self.volumes[self.begin_index: self.end_index \
                                                    + future_margin]
            plt.plot(
                indexes[index:], volumes[index:], label='log(volume orig.)')
        
        plt.legend()
        plt.show()

    def plot_prediction(
            self, dt, predictions, future=None, 
            show_features=True, data_count=50):
        """
        Parameters:
        -----------

        dt : ``DataTransfer`` object.
        """

        # first_index = indexes_tf[0]
        # index = np.where(dt.indexes > first_index)[0][0] - 1
        index = 0

        if dt.opens is not None:
            plt.plot(dt.get_indexes()[index:], dt.opens[index:], 
                     label='open orig.')
        if dt.volumes is not None:
            plt.plot(dt.get_indexes()[index:], dt.volumes[index:], 
                     label='volume orig.')
            
        data_count = min(len(dt.indexes_tf), data_count)

        targets = dt.targets.transpose()
        for i in range(len(targets)):
            plt.plot(
                dt.indexes_tf + self.future_count, 
                targets[i], linewidth=7, alpha=0.5, # marker='o', 
                     label=self.target_names()[i])
        if show_features:
            features = dt.features.transpose()
            plt.plot(
                dt.indexes_tf, 
                features[-1], # marker='x', 
                        label='features[-1]')
        plt.vlines(
            dt.indexes_tf[-1], np.min(targets[0]), np.max(targets[0]), 
            linestyle='dashed',
            label='time now')
        # import pdb; pdb.set_trace()
        plt.plot(
            predictions[0] + self.future_count, 
            predictions[1],  
            linewidth=5, alpha=0.5, label='prediction')
        
        if future is not None:
            plt.plot(
                future[0], 
                future[1], 
                label='future')            
        
        plt.legend()
        plt.show()