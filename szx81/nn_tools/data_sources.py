import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import core as co
import hist_data as hd
hd.set_hist_data(data_count=None)

from nn_tools.data_source_super import DataSource

class OpenVolumeDs(DataSource):
    def fit_data(self, **kwargs):
        self.step = kwargs['step']
        opens = []
        volumes = []

        for i in range(0, len(self.data)):  
            open = 0
            volume = 0
            if not i + self.step < len(self.data):
                break
            for k in range(self.step):
                val = self.data[i + k]
                open += (val[1][0][0] + val[1][1][0]) / 2
                volume += val[2]              

            opens.append(open / self.step) #.reshape(-1, 1)
            volumes.append(volume / self.step) #.reshape(-1, 1)

        opens = np.array(opens).reshape(-1, 1)
        volumes = np.log(np.array(volumes).reshape(-1, 1) + 10.0)

        self.fit_transform([opens, volumes])

    def feature_names(self):
        return ('opens', 'volumes')
    
    def target_names(self):
        return ('opens target',)
        
    def get_data__(self):
        def helper(i):
            val = self.data[i]
            return (val[1][0][0] + val[1][1][0]) / 2, val[2]

        indexes = [] 
        opens = []
        volumes = []
        targets = []
        for i in range(self.begin, self.end_index, self.step):
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
                    indexes.append(index)
                    opens.append(open / self.step)
                    volumes.append(volume / self.step)
                    targets.append(target / self.step)

        opens = np.array(opens).reshape(-1, 1)
        volumes = np.log(np.array(volumes).reshape(-1, 1) + 10.0)
        opens, volumes = self.transform([opens, volumes])

        targets = np.array(targets).reshape(-1, 1)
        targets = self.transform(targets, index=0)

        self.features = np.concatenate((opens, volumes), axis=1)
        self.targets = targets
        self.indexes = np.array(indexes).reshape(-1, 1)
        return (
            self.features, 
            self.targets, 
            self.indexes)


class OpenDs(DataSource):
    def fit_data(self, **kwargs):
        self.step = kwargs['step']
        opens = []
        for i in range(0, len(self.data)):  
            open = 0
            if not i + self.step < len(self.data):
                break
            for k in range(self.step):
                val = self.data[i + k]
                open += (val[1][0][0] + val[1][1][0]) / 2           

            opens.append(open / self.step) #.reshape(-1, 1)

        opens = np.array(opens).reshape(-1, 1)

        self.fit_transform([opens,])

    def feature_names(self):
        return ('opens',)
    
    def target_names(self):
        return ('opens target',)
        
    def get_data__(self):
        def helper(i):
            val = self.data[i]
            return (val[1][0][0] + val[1][1][0]) / 2

        indexes = [] 
        opens = []
        targets = []
        for i in range(self.begin, self.end_index, self.step):
            index = 0
            open = 0
            target = 0
            
            for k in range(self.step):
                if k == self.step - 1:
                    _open = helper(i + k)
                    open += _open
                    _open = helper(i + k + self.future_count)
                    target += _open
                    index = i + k                    
                    indexes.append(index)
                    opens.append(open / self.step)
                    targets.append(target / self.step)

        opens = np.array(opens).reshape(-1, 1)
        opens = self.transform([opens,])

        targets = np.array(targets).reshape(-1, 1)
        targets = self.transform(targets, index=0)

        self.features = np.concatenate((opens,), axis=1)
        self.targets = targets
        self.indexes = np.array(indexes).reshape(-1, 1)
        return (
            self.features, 
            self.targets, 
            self.indexes)

def test_ds():
    # ds = OpenVolumeDs(
    #     hd.DICT_DATA.values(), (StandardScaler(), StandardScaler()), step=10)
    ds = OpenDs(
        hd.DICT_DATA.values(), (MinMaxScaler(), MinMaxScaler()), step=10)
    
    features, targets, indexes = ds.get_data(
        end_index=5000, data_count=3000, future_count=3)
    ds.report()
    ds.plot()

    print(f'''
    wanted end range: ({1000})
    index range: ({indexes[0]}, {indexes[-1]})
    targets.shape: {targets.shape}
    indexes.shape: {indexes.shape}
    features.shape: {features.shape}
    ''')

class ContextSequencer:
    def __init__(self, 
                data_source, 
                seq_len=5,  
                future_len=5, 
                end_day=0):
        self.data_source = data_source
        self.seq_len = seq_len
        self.future_len = future_len
        self.first_trained_index = end_day * co.config.PERIOD * 60 * 24
        self.trained_indexes = set()
        self.last_trained_index = None

    def create_sequences(self, end_index, seq_len, data_count):
        """Lists sequences of ``data`` items, ``seq_len`` long, ``data_count`` 
        of them, ending - not including - ``end_index`` index of ``data``. Each next sequence is shifted by 1 from the previous.
        """        

        step = self.data_source.step
        _features, _targets, _indexes = self.data_source.get_data(
                            end_index, 
                            data_count + seq_len * step + self.future_len)
        features = []
        targets = []
        indexes = []
        for i in range(data_count):
            features.append(
                _features[i: (i + seq_len)].flatten()
            )
            indexes.append(
                _indexes[i + seq_len + self.future_len - 1]
            )
            targets.append(
                _targets[i + seq_len + self.future_len - 1]
            )

        features = np.array(features)
        targets = np.array(targets)
        indexes = np.array(indexes) 

        return (
            features[~np.isnan(features)], 
            targets[~np.isnan(targets)], 
            indexes[~np.isnan(indexes)]
            )

def test_cs():
    cs = ContextSequencer(
    OpenVolumeDs(
        hd.DICT_DATA.values(), 
        (StandardScaler(), StandardScaler()),
        step=3
        ))

    features, targets, indexes = cs.create_sequences(
        end_index=30000, seq_len=10, data_count=25000)
    print(features[:10])

def main():
    # test_debug()
    test_ds()
    
if __name__ == "__main__":
    main()  