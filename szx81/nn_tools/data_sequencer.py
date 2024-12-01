import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
np.random.seed(0)

import core as co
import hist_data as hd

DAY = 60 * 24
FUTURE = 5
TRAINING_NUM = 2 * DAY
TESTING_NUM = 1 * DAY
SEQ_LEN = 10

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
    
    def get_data(self, debug=False):
        if debug:
           data_x = [i for i in range(len(self.data_x))]
           return data_x, data_x.copy()

        return self.data_x, self.data_y

class ContextSequencer:
    def __init__(self, 
                data_source_class, 
                seq_len=SEQ_LEN, 
                future_len=FUTURE, start_day=0):
        self.data_source = data_source_class.get_instance()
        self.seq_len = seq_len
        self.future_len = future_len
        self.start_index = start_day // DAY
        self.testing_num = 0

    def __create_sequences(self, start, seq_len, num, data_x, data_y):
        """Lists sequences of ``data`` items, ``seq_len`` long, ``num`` of them, 
        starting from ``start`` index of ``data``. Each next sequence is shifted
        by 1 from the previous.
        """
        list_x = []
        list_y = []
        for i in range(start, start + num):
            x = data_x[i: (i + seq_len)]
            list_x.append(x)

        for i in range(start + self.future_len - 1, start + num + self.future_len - 1):
            y = data_y[i + seq_len]
            list_y.append(y)

        return np.array(list_x), np.array(list_y)
    
    def get_sequences(self, num, debug=False):

        data_len = num + self.future_len + self.seq_len
        if self.data_source is None or self.start_index < 0 \
                or self.start_index + data_len > self.data_source.len():
            raise Exception('data_source is None or start_index < 0 ' \
                        + 'or start_index + data_len > data_source()')
            
        data_x, data_y = self.data_source.get_data(debug)
        
        x, y = self.__create_sequences(
                self.start_index, 
                self.seq_len, num, data_x, data_y)
        
        self.start_index = self.start_index + data_len

        if debug:
            title = ''
            title += f'start={self.start_index} '
            title += f'data len={data_len}, seq_len={self.seq_len}, num={num} '
            title += f'future_len={self.future_len}'
            print(title)
            print(f'x:\n{x}')
            print(f'y:\n{y}')

        return x, y

def test_debug():
    cs = ContextSequencer(ForexDataSource, start_day=0, seq_len=3, future_len=2)
    x_train, y_train = cs.get_sequences(num=3, debug=True)
    x_test, y_test = cs.get_sequences(num=3, debug=True)

def test():
    x_train, y_train, x_test, y_test = ContextSequencer(ForexDataSource).get_train_test_data(ForexDataSource)

def main():
    test_debug()

if __name__ == "__main__":
    main()  

