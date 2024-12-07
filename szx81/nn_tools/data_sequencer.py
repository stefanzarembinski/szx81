import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
np.random.seed(0)

import matplotlib.pyplot as plt

import nn_tools.data_source as ds

DAY = 60 * 24
FUTURE = 5
TRAINING_NUM = 2 * DAY
TESTING_NUM = 1 * DAY
SEQ_LEN = 10

class ContextSequencer:
    # _instance = None
    # def __new__(cls, 
    #             data_source_class, 
    #             seq_len=SEQ_LEN,  
    #             future_len=FUTURE, end_day=0):
    #     if not cls._instance:
    #         cls._instance = super(ContextSequencer, cls).__new__(cls)
    #     return cls._instance 

    def __init__(self, 
                data_source, 
                seq_len=SEQ_LEN,  
                future_len=FUTURE, 
                end_day=0):
        self.data_source = data_source
        self.seq_len = seq_len
        self.future_len = future_len
        self.first_trained_index = end_day * DAY
        self.trained_indexes = set()
        self.last_trained_index = None
 
    def create_sequences(self, end_index, seq_len, count, debug=False):
        """Lists sequences of ``data`` items, ``seq_len`` long, ``count`` 
        of them, ending - not including - ``end_index`` index of ``data``. Each next sequence is shifted by 1 from the previous.
        """
        data_len = count + self.future_len + self.seq_len

        if self.data_source is None or self.first_trained_index < 0 \
                or (self.data_source.len() > 0) \
                    and self.first_trained_index + data_len > self.data_source.len():
            raise Exception('data_source is None or end_index < 0 ' \
                        + 'or end_index + data_len > data_source()')
        
        data, indexes = self.data_source.get_data(
                            end_index, count + seq_len + self.future_len, debug)
        list_x = []
        list_y = []
        x_indexes = []
        y_indexes = []
        for i in range(count):
            list_x.append(data[i: (i + seq_len)])
            x_indexes.append(indexes[i: (i + seq_len)])
            list_y.append(data[i + seq_len + self.future_len - 1])
            y_indexes.append(indexes[i + seq_len + self.future_len - 1])
        return np.array(list_x), np.array(list_y), x_indexes, y_indexes
    
    def get_sequences(self, count, debug=False):
        x, y, indexes_x, indexes_y = self.create_sequences(
                        self.first_trained_index, self.seq_len, count, debug)
        retval = x, y, indexes_x, indexes_y
        data_len = count + self.future_len + self.seq_len
        self.last_trained_index = self.first_trained_index + data_len

        if debug:
            title = ''
            title += f'end={self.first_trained_index} '
            title += f'data len={data_len}, seq_len={self.seq_len}, count={count} '
            title += f'future_len={self.future_len}'
            print(title)
            print(f'x:\n{x}')
            print(f'y:\n{y}')

            retval = x, y, self.data_source.get_data(
                                self.first_trained_index, count + self.seq_len)

        return retval
    
    def get_training(self, count, verbose=True):
        x, y, indexes_x, indexes_y = self.get_sequences(count)
        
        self.trained_indexes = self.trained_indexes | set(indexes_y)
        if verbose:
            print(f'begin index: {self.first_trained_index}, end index: {self.last_trained_index}, count:{count}')
        return x, y
    
    def get_testing(self, context_count, dist_count, test_count, verbose=True):
        """Returns test data ``count`` long, beginning ``dist_count``after  training end. The data have prepended ``context_count`` long 
        'warming-up' part.
        """
        count = context_count + test_count
        end_index = self.last_trained_index + dist_count + test_count
        begin_index = end_index - count
        if verbose:
            print(f'test count: {test_count}, begin index: {begin_index}, end index: {end_index}')

        x, y, indexes  = self.create_sequences(
                                end_index, self.seq_len, count)
        if len(self.trained_indexes & set(indexes)) > 0:
            raise Exception('Testing data include trained parts!')
            
        return  x, y
    
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

def main():
    # test_debug()
    test()
    

if __name__ == "__main__":
    main()  

