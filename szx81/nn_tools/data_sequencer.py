import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
np.random.seed(0)

import matplotlib.pyplot as plt
import nn_tools.data_source as ds
import logging
# logging.basicConfig(format="{levelname}:{funcName}: {message}", style="{")
log = logging.getLogger(__file__)
log.setLevel(level=logging.DEBUG)

import core as co

DAY = 60 * 24
FUTURE = 5
TRAINING_NUM = 2 * DAY
TESTING_NUM = 1 * DAY
SEQ_LEN = 10

# def set_log_level(level=logging.DEBUG):
#     global log
#     log.setLevel(level)

class ContextSequencer:
    def __init__(self, 
                data_source, 
                seq_len,
                future_count=5,
                end_index=5000):
        self.data_source = data_source
        self.seq_len = seq_len
        self.future_count = future_count
        self.end_index = end_index
        self.trained_indexes = set()
        self.data_source.future_count = future_count
    
    def create_sequences(self, end_index, data_count):
            """Lists sequences of ``data`` items, ``seq_len`` long, ``data_count`` 
            of them, ending - not including - ``end_index`` index of ``data``. Each next sequence is shifted by 1 from the previous.
            """        
            dt = self.data_source.get_data(
                                            end_index, 
                                            data_count,
                                            self.future_count)
            features = []
            targets = []
            indexes_tf = []
            for i in range(data_count - self.future_count):
                features.append(dt.features[i: (i + self.seq_len)].flatten())
                indexes_tf.append(
                    dt.indexes_tf[i + self.seq_len])
                targets.append(
                    dt.targets[i + self.seq_len + self.future_count - 1])

            log.debug(f'''
    len(features): {len(features)}
    len(features[i]): {len(features[0])}
    features[0]: {features[0]}
        features[1]: {features[1]}
            features[2]: {features[2]}
                ...
    len(targets): {len(targets)}
    targets[0:3]: {targets[0:3]}
        targets[1:4]: {targets[1:4]}
            targets[2:5]: {targets[2:5]}
                ...
    len(indexes): {len(indexes_tf)}
    indexes[0:3]: {indexes_tf[0:3]}
        indexes[1:4]: {indexes_tf[1:4]}
            indexes[2:5]: {indexes_tf[2:5]}
                ...
    ''')
            dt.features = np.array(features)
            dt.targets = np.array(targets)
            dt.indexes_tf = np.array(indexes_tf)
            return dt
    
    def get_training(self, data_count, end_index=None, verbose=True):
        if end_index is None:
            end_index = self.end_index
        
        dt = self.create_sequences(
                                end_index=end_index,
                                data_count=data_count
                                )
        
        self.trained_indexes = self.trained_indexes | set(dt.indexes_tf)
        if verbose:
            print(f'''
TRAINING DATA
begin index: {dt.indexes_tf[0]}, 
end index: {dt.indexes_tf[-1]}, 
data count: {data_count}
sequence length: {self.seq_len}
''')
        
        return dt
    
    def get_testing(self, context_count, dist_count, data_count):
        """Returns test data ``count`` long, beginning ``dist_count``after  training end. The data have prepended ``context_count`` long 
        'warming-up' part.
        """
        
        begin_index = max(self.trained_indexes) + dist_count
        index_count = context_count * self.seq_len + data_count
        end_index = begin_index + index_count

        dt = self.create_sequences(
                                end_index=end_index,
                                data_count=context_count + data_count
                                )
        
        if len(self.trained_indexes & set(dt.indexes_tf)) > 0:
            raise Exception('Testing data include trained parts!')
        
        return dt 
    
    def plot(self):
        count = 10
        x, y, (data, indexes) = self.get_sequences(count=count, debug=True)

        _, ax1 = plt.subplots()
        ax1.plot(indexes, data, color='black', label='data')
        ax1.set_ylabel('data', color='black')

        ax2 = ax1.twinx()
        ax1.set_ylabel('context sequences', color='blue')
        first = True
        for i in range(count):
            if first:
                first = False
                ax2.scatter(
                    x[i], [i] * len(x[i]), label='context', color='blue')
                ax2.scatter(y[i], [i], label='target', color='orange')
            ax2.scatter(x[i], [i] * len(x[i]), color='blue')
            ax2.scatter(y[i], [i], color='orange')            

        plt.legend()
        plt.show()

def test_debug():
    cs = ContextSequencer(
        ds.ForexDataSource, end_day=1, seq_len=10, future_len=2)
    cs.plot()

def test():
    end_day = 2
    context_len = 10
    future_len = 5

    cs = ContextSequencer(
        ds.ForexDataSource, end_day=end_day, seq_len=context_len, 
        future_len=future_len)
    x, y = cs.get_sequences(count=1000)

    print(f'x:\n{x[-1]}')
    print(f'y:\n{y[-1]}')
    print(f'end_index: {cs.end_index}')

def test_cs():
    import random
    import math
    from sklearn.preprocessing import MinMaxScaler
    import hist_data as hd
    from nn_tools.data_sources import OpenDs, SinusDs, set_logging_level

    # set_logging_level()

    # cs = ContextSequencer(
    #     data_source=OpenDs(
    #         hd.DICT_DATA.values(), 
    #         (MinMaxScaler(), MinMaxScaler()),
    #         step=3
    #         ),
    #     seq_len=5,
    #     future_count=10,
    #     end_index=500
    #     )
    
    sinus = lambda j, noise: .5 * math.sin(j * .03) + random.uniform(
        -noise, noise) + .5
    cs = ContextSequencer(
        data_source=SinusDs(
            sinus, 
            (None, None),
            step=3,
            noise=0.03
            ),
        seq_len=5,
        future_count=10,
        end_index=500
        )

    dt = cs.create_sequences(
        end_index=300, data_count=1500)
    print(f'''
features: 
{dt.features}
targets:
{dt.targets}
indexes:
{dt.indexes_tf}
''')


def main():
    # test_debug()
    # test()
    # set_log_level(logging.DEBUG)
    test_cs()
    

if __name__ == "__main__":
    main()  

