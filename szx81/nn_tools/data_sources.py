import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

import logging as lg
# https://realpython.com/python-logging/
# https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file

# lg.basicConfig(format="{levelname}:{name}:{message}", style="{")
lg.basicConfig(format="{levelname}:{funcName}: {message}", style="{")
log = lg.getLogger(__name__)
log.setLevel(level=lg.DEBUG)

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

            opens.append(open / self.step)
            volumes.append(volume / self.step)

        opens = np.array(opens)
        volumes = np.log(np.array(volumes) + 10.0)

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
        # for i in range(len(self.indexes_hist)):
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
        #             indexes.append(index)
        #             opens.append(open / self.step)
        #             volumes.append(volume / self.step)
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
                    indexes.append(index)
                    opens.append(open / self.step)
                    volumes.append(volume / self.step)
                    targets.append(target / self.step)

        opens = np.array(opens)
        volumes = self.log_volume(np.array(volumes))
        opens, volumes = self.transform([opens, volumes])

        targets = np.array(targets)
        targets = self.transform(targets, index=0)

        self.features = np.concatenate((opens, volumes), axis=1)
        self.targets = targets
        self.indexes = np.array(indexes)
        return (
            self.features, 
            self.targets, 
            self.indexes)

class OpenDs(DataSource):

    def fit_data(self):
        self.step = self.kwargs['step']
        opens = []
        for i in range(0, len(self.data)):  
            open = 0
            if not i + self.step < len(self.data): 
                break
            for k in range(self.step):
                val = self.data[i + k]
                open += (val[1][0][0] + val[1][1][0]) / 2           

            opens.append(open / self.step)

        opens = np.array(opens)

        log.debug(f'''
before transformation
len(self.data): {len(self.data)}
len(opens): {len(opens)}
opens(min. max) = ({np.min(opens):.1e}, {np.max(opens):.1e})
''')

        opens, = self.fit_transform([opens,])

        log.debug(f'''
after transformation
len(self.opens): {len(opens)}                  
opens(min, max): ({np.min(opens):.1e}, {np.max(opens):.1e})''')


    def feature_names(self):
        return ('opens',)
    
    def target_names(self):
        return ('opens target',)
    
    def raw_data_begin(self):
        self.feature_data_count = self.data_count * self.step
        return self.end_index \
            - (self.feature_data_count + (self.future_count + self.step))
        
    def get_data__(self):
        indexes = [] 
        opens = []
        targets = []
        opens = []
        
        for i in range(self.feature_data_count):
            open = 0
            target = 0

            for k in range(self.step):
                open += self.opens_hist[i + k]
                target += self.opens_hist[i + k + self.future_count]
                if k == self.step - 1:
                    indexes.append(self.indexes_hist[i + k])
                    opens.append(open / self.step)
                    targets.append(target / self.step)

        indexes = np.array(indexes)
        opens = np.array(opens)
        opens = self.transform(opens, 0)
        targets = np.array(targets)
        targets = self.transform(targets, 0)

        log.debug(f'''
len(indexes): {len(indexes)}
indexes(min. max): ({np.min(indexes)}, {np.max(indexes)})
len(opens): {len(opens)}
opens(min. max): ({np.min(opens):.1e}, {np.max(opens):.1e})
len(targets): {len(targets)}
targets(min. max): ({np.min(targets):.1e}, {np.max(targets):.1e})
''')
        self.features = np.concatenate((opens,), axis=1)
        self.targets = targets
        self.indexes = indexes
        return (
            self.features, 
            self.targets, 
            self.indexes)
    
    def plot_ds(self, plt, data_count=10, show=False):
        targets = self.targets.transpose()
        for i in range(len(targets)):
            plt.plot(
                self.indexes[-data_count:], 
                targets[i][-data_count:], # marker='o', 
                     label=self.target_names()[i])
        features = self.features.transpose()
        for i in range(len(features)):
            plt.plot(
                self.indexes[-data_count:], 
                features[i][-data_count:], # marker='x', 
                     label=self.feature_names()[i])
        if show:
            plt.legend()
            plt.show()
        
        return self.indexes[-data_count:][0]

def test_ds():
    # ds = OpenVolumeDs(
    #     hd.DICT_DATA.values(), (StandardScaler(), StandardScaler()), step=10)
    ds = OpenDs(
        hd.DICT_DATA.values(), (MinMaxScaler(), MinMaxScaler()), step=3)
    
    ds.get_data(end_index=5000, data_count=30, future_count=3)
    ds.report()
    # ds.plot_ds(plt, data_count=10, show=True)
    ds.plot()


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