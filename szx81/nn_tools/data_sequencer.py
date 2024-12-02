import random
import math
import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
np.random.seed(0)

import matplotlib.pyplot as plt

import core as co
import hist_data as hd

DAY = 60 * 24
FUTURE = 5
TRAINING_NUM = 2 * DAY
TESTING_NUM = 1 * DAY
SEQ_LEN = 10

class SinusDataSource:
    __instance = None
    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance
    
    def __init__(self):
        pass        

    def len(self):
        return -1
    
    @staticmethod
    def sinus_data(begin, count):
        return [math.sin(i * .1) + random.uniform(-.5, .5) for i in range(begin, begin + count + 1)]

    def get_data(self, end_index, count, debug=False):
        begin = end_index - count - 1
        end = end_index 
        indexes = [i for i in range(begin, end)]      
        if debug:
           return indexes, indexes
        return self.sinus_data(begin, count), indexes

class ForexDataSource:
    __instance = None
    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance
    
    def __init__(self):
        hd.set_hist_data(data_count=None)
        hd_values = list(hd.DICT_DATA.values())
        self.data_x = []
        self.data_y = []
        for val in hd_values:
            y = (val[1][0][0] + val[1][1][0]) / 2
            self.data_y.append(y)
            # self.data_x.append((y, hd_values[i][2]))
            self.data_x.append(y)        

    def len(self):
        return len(self.data_x)
    
    def get_data(self, end_index, count, debug=False):
        begin = end_index - count - 1
        end = end_index 
        indexes = [i for i in range(begin, end)]      
        if debug:
           return indexes, indexes
        return self.data_x[begin: end], indexes

class ContextSequencer:
    def __init__(self, 
                data_source_class, 
                seq_len=SEQ_LEN,  
                future_len=FUTURE, end_day=0):
        self.data_source = data_source_class.get_instance()
        self.seq_len = seq_len
        self.future_len = future_len
        self.end_index = end_day * DAY
        self.testing_count = 0

    def __create_sequences(self, end_index, seq_len, count, debug=False):
        """Lists sequences of ``data`` items, ``seq_len`` long, ``count`` 
        of them, starting from ``end_index`` index of ``data``. Each next sequence 
        is shifted by 1 from the previous.
        """
        data, _ = self.data_source.get_data(
                                end_index, count + seq_len, debug)
        list_x = []
        list_y = []
        for i in range(count):
            x = data[i: (i + seq_len)]
            list_x.append(x)
            list_y.append(data[i + seq_len + self.future_len - 1])
        # import pdb; pdb.set_trace()
        return np.array(list_x), np.array(list_y)
    
    def get_sequences(self, count, debug=False):

        data_len = count + self.future_len + self.seq_len
        if self.data_source is None or self.end_index < 0 \
                or (self.data_source.len() > 0) \
                    and self.end_index + data_len > self.data_source.len():
            raise Exception('data_source is None or end_index < 0 ' \
                        + 'or end_index + data_len > data_source()')
        
        x, y = self.__create_sequences(
                                self.end_index, self.seq_len, count, debug)
        retval = x, y

        if debug:
            title = ''
            title += f'end={self.end_index} '
            title += f'data len={data_len}, seq_len={self.seq_len}, count={count} '
            title += f'future_len={self.future_len}'
            print(title)
            print(f'x:\n{x}')
            print(f'y:\n{y}')

        if debug:
            retval = x, y, self.data_source.get_data(
                                        self.end_index, count + self.seq_len)
            
        self.end_index = self.end_index + data_len

        return retval
    
    @classmethod
    def plot(cls, data_source_class):
        seq_len = 3
        future_len = 2
        count = 10
        cs = cls(
            data_source_class, end_day=10, 
            seq_len=seq_len, future_len=future_len)
        x, y, (data, indexes) = cs.get_sequences(count=count, debug=True)

        _, ax1 = plt.subplots()
        ax1.plot(indexes, data, color='black', label='data')
        ax1.set_ylabel('data', color='black')

        ax2 = ax1.twinx()
        ax1.set_ylabel('context sequences', color='blue')
        for i in range(count):
            # import pdb; pdb.set_trace()
            ax2.scatter(x[i], [i] * len(x[i]), color='blue')
            ax2.scatter(y[i], [i], color='orange')

        plt.legend()
        plt.show()


def test_debug():
    cs = ContextSequencer(
        SinusDataSource, end_day=1, seq_len=10, future_len=2)
    ContextSequencer.plot(SinusDataSource)

def test():
    x_train, y_train, x_test, y_test = ContextSequencer(ForexDataSource).get_train_test_data(ForexDataSource)

def main():
    test_debug()

if __name__ == "__main__":
    main()  

