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

# class SinusDataSource:
#     __instance = None
#     @classmethod
#     def get_instance(cls):
#         if cls.__instance is None:
#             cls.__instance = cls()
#         return cls.__instance
    
#     def __init__(self):
#         hd.set_hist_data(data_count=None)
#         hd_values = list(hd.DICT_DATA.values())
#         self.data_x = []
#         self.data_y = []
#         for val in hd_values:
#             y = (val[1][0][0] + val[1][1][0]) / 2
#             self.data_y.append(y)
#             # self.data_x.append((y, hd_values[i][2]))
#             self.data_x.append(y)        

#     def len(self):
#         return len(self.data_x)
    
#     def get_data(self, debug=False):
#         if debug:
#            data_x = [i for i in range(len(self.data_x))]
#            return data_x, data_x.copy()
#         return np.sin([i * .1 for i in range(count)])

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
    
    def get_data(self, start, count, future, debug=False):
        # import pdb; pdb.set_trace()
        if count > 0:
            begin_x = start
            end_x = start + count
            begin_y = start + future
            end_y = start + count + future
        else: 
            begin_x = start + count
            end_x = start
            begin_y = start + count - future
            end_y = start            
        if debug:
           data_x = [i for i in range(begin_x, end_x)]
           data_y = [i for i in range(begin_y, end_y)]
           return data_x, data_y

        return self.data_x[begin_x: end_x], self.data_y[begin_y: end_y]

class ContextSequencer:
    def __init__(self, 
                data_source_class, 
                seq_len=SEQ_LEN, 
                future_len=FUTURE, start_day=0):
        self.data_source = data_source_class.get_instance()
        self.seq_len = seq_len
        self.future_len = future_len
        self.start_index = start_day // DAY
        self.testing_count = 0

    def __create_sequences(self, start, seq_len, count, debug=False):
        """Lists sequences of ``data`` items, ``seq_len`` long, ``count`` 
        of them, starting from ``start`` index of ``data``. Each next sequence 
        is shifted by 1 from the previous.
        """
        data_x, data_y = self.data_source.get_data(
                                        start, count, self.future_len, debug)
        
        list_x = []
        list_y = []
        for i in range(start, start + count):
            x = data_x[i: (i + seq_len)]
            list_x.append(x)

        for i in range(
                start + self.future_len, 
                start + count + self.future_len):
            y = data_y[i]
            list_y.append(y)

        return np.array(list_x), np.array(list_y)
    
    def get_sequences(self, count, debug=False):

        data_len = count + self.future_len + self.seq_len
        if self.data_source is None or self.start_index < 0 \
                or self.start_index + data_len > self.data_source.len():
            raise Exception('data_source is None or start_index < 0 ' \
                        + 'or start_index + data_len > data_source()')
        
        x, y = self.__create_sequences(
                                self.start_index, self.seq_len, count, debug)
        
        self.start_index = self.start_index + data_len

        if debug:
            title = ''
            title += f'start={self.start_index} '
            title += f'data len={data_len}, seq_len={self.seq_len}, count={count} '
            title += f'future_len={self.future_len}'
            print(title)
            print(f'x:\n{x}')
            print(f'y:\n{y}')

        return x, y
    
    @classmethod
    def plot(cls):
        seq_len = 3
        future_len = 2
        count = 3
        cs = cls(
            ForexDataSource, start_day=0, 
            seq_len=seq_len, future_len=future_len)
        x, y = cs.get_sequences(count=count, debug=True)
        for i in range(count):
            # import pdb; pdb.set_trace()
            plt.scatter(x[i], [i] * len(x[i]), color='blue')
            plt.scatter(y[i], [i], color='orange')

        plt.show()


def test_debug():
    cs = ContextSequencer(ForexDataSource, start_day=0, seq_len=3, future_len=2)
    x_train, y_train = cs.get_sequences(count=3, debug=True)
    x_test, y_test = cs.get_sequences(count=3, debug=True)

    ContextSequencer.plot()

def test():
    x_train, y_train, x_test, y_test = ContextSequencer(ForexDataSource).get_train_test_data(ForexDataSource)

def main():
    test_debug()

if __name__ == "__main__":
    main()  

