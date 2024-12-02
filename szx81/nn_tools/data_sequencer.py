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
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(SinusDataSource, cls).__new__(cls)
        return cls._instance    

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
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ForexDataSource, cls).__new__(cls)
        return cls._instance  
    
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
    # _instance = None
    # def __new__(cls, 
    #             data_source_class, 
    #             seq_len=SEQ_LEN,  
    #             future_len=FUTURE, end_day=0):
    #     if not cls._instance:
    #         cls._instance = super(ContextSequencer, cls).__new__(cls)
    #     return cls._instance 

    def __init__(self, 
                data_source_class, 
                seq_len=SEQ_LEN,  
                future_len=FUTURE, end_day=0):
        self.data_source = data_source_class()
        self.seq_len = seq_len
        self.future_len = future_len
        self.end_index = end_day * DAY
        self.training_end = None

    def __create_sequences(self, end_index, seq_len, count, debug=False):
        """Lists sequences of ``data`` items, ``seq_len`` long, ``count`` 
        of them, starting from ``end_index`` index of ``data``. Each next sequence 
        is shifted by 1 from the previous.
        """
        data_len = count + self.future_len + self.seq_len
        if self.data_source is None or self.end_index < 0 \
                or (self.data_source.len() > 0) \
                    and self.end_index + data_len > self.data_source.len():
            raise Exception('data_source is None or end_index < 0 ' \
                        + 'or end_index + data_len > data_source()')
        
        data, _ = self.data_source.get_data(
                            end_index, count + seq_len + self.future_len, debug)
        list_x = []
        list_y = []
        for i in range(count):
            x = data[i: (i + seq_len)]
            list_x.append(x)
            list_y.append(data[i + seq_len + self.future_len - 1])
        # import pdb; pdb.set_trace()
        return np.array(list_x), np.array(list_y)
    
    def get_sequences(self, count, debug=False):
        x, y = self.__create_sequences(
                                self.end_index, self.seq_len, count, debug)
        retval = x, y
        data_len = count + self.future_len + self.seq_len

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
    
    def get_training(self, count, verbose=True):
        self.training_end = self.end_index
        begin_index = self.end_index
        retval = self.get_sequences(count)
        if verbose:
            print(f'begin index: {begin_index}, end index: {self.end_index}, count:{count}')
        return retval
    
    def get_testing(self, context_count, dist_count, verbose=True):
        end_index = self.training_end + dist_count
        begin_index = end_index - context_count
        if verbose:
            print(f'begin index: {begin_index}, end index: {end_index}')        
        return self.__create_sequences(
                                end_index, self.seq_len, context_count)
    
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
        ForexDataSource, end_day=1, seq_len=10, future_len=2)
    cs.plot()

def test():
    end_day = 2
    context_len = 10
    future_len = 5

    cs = ContextSequencer(
        ForexDataSource, end_day=end_day, seq_len=context_len, 
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

