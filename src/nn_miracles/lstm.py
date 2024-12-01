import sys
import os
from os import path

import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
np.random.seed(0)

SRC_DIR = path.join(path.dirname(__file__), '../')
sys.path.append(SRC_DIR)

import core as co
import hist_data as hd

hd.set_hist_data(data_count=None)
hd_values = list(hd.DICT_DATA.values())

DAY = 60 * 24
FUTURE = 5
TRAINING_NUM = 2 * DAY
TESTING_NUM = 1 * DAY
SEQ_LEN = 10

def forex_data_source(index=None, total_len=None, debug=False):
    if index is None:
        return len(hd_values)
    
    data_X = []
    data_y = []
    for i in range(index, index + total_len):
        y = (hd_values[i][1][0][0] + hd_values[i][1][1][0]) / 2
        if debug:
            y = i
        data_y.append(y)
        # data_X.append((y, hd_values[i][2]))
        data_X.append(y)

    return data_X, data_y

def train_test_data(
        data_source,
        training=TRAINING_NUM, testing=TESTING_NUM, 
        seq_len=SEQ_LEN, future=FUTURE, start=0, debug=False):
    
    X_train = []
    y_train = []

    X_test = []
    y_test = []
    index = max(0, start // DAY)
    total_len = training + testing + future + seq_len
    
    if data_source() is not None:
        if index + total_len > data_source():
            raise Exception('')
    
    data_X, data_y = data_source(index, total_len, debug)
    
    def create_sequences(start, seq_len, num):
        """Lists sequences of ``data`` items, ``seq_len`` long, ``num`` of them, 
        starting from ``start`` index of ``data``. Each next sequence is shifted
        by 1 from the previous.
        """
        list_x = []
        list_y = []
        for i in range(start, start + num):
            x = data_X[i: (i + seq_len)]
            list_x.append(x)

        for i in range(start + future - 1, start + num + future - 1):
            y = data_y[i + seq_len]
            list_y.append(y)

        return np.array(list_x), np.array(list_y)

    X_train, y_train = create_sequences(index, seq_len, training)
    X_test, y_test = create_sequences(index + training, seq_len, testing)

    if debug:
        print(f'start={start}, training={training}, testing={testing}, seq_len={seq_len}, future={future}')
        print(f'X_train:\n{X_train}')
        print(f'y_train:\n{y_train}')
        print()
        print(f'X_test:\n{X_test}')
        print(f'y_test:\n{y_test}')

    return(X_train, y_train, X_test, y_test)

# __ = train_test_data(forex_data_source, 3, 3, 3, debug=True)
# __ = train_test_data(sinus_data_source, 3, 3, 3, debug=True)

def test_debug():
    X_train, y_train, X_test, y_test = train_test_data(forex_data_source, 
        start=0, training=3, testing=3, seq_len=3, future=2, debug=True)

def test():
    X_train, y_train, X_test, y_test = train_test_data(forex_data_source)

def main():
    test_debug()

if __name__ == "__main__":
    main()  

