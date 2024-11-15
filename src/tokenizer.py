import collections
from os import path
import math
import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import core as co
from core import config
import piecewise_fit as ls
import test_data as td

class Tokenizer:
    number_pieces = 10
    window = 120
    margin = 0.05
    time_qizer = None
    value_qizer = None
    temperature_qizer = None
    filter = co.Savgol_filter(window=50, order=5)
    none_word = '0000'

    def __init__(self, data):
        self.data = data
        self.clazz = None
        self.time_part = None
        self.value_part = None
        self.temp_part = None

    def save_story(whole_story):
        whole_story = Tokenizer.get_sentence_str(whole_story)
        with open(
            path.join(co.DATA_STORE, f'whole_story_{Tokenizer.window}_{Tokenizer.number_pieces}.txt'), 'w') as f:
            for word in whole_story:
                f.write(f'{str(word)}\n')

    def get_story_from_file():
        words = []
        with open(
            path.join(
                co.DATA_STORE, 
                f'whole_story_{Tokenizer.window}_{Tokenizer.number_pieces}.txt'), 'r') as f:
            for line in f:
                words.append(line[:-1])
        return words

    def save_words(words):
        with open(
            path.join(
                co.DATA_STORE, 
                f'words_{Tokenizer.window}_{Tokenizer.number_pieces}.txt'), 'w') as f:
            for word in words:
                f.write(f'{str(word)}\n')

    def get_words_from_file():
        words = []
        with open(
            path.join(
                co.DATA_STORE, 
                f'words_{Tokenizer.window}_{Tokenizer.number_pieces}.txt'), 'r') as f:
            for line in f:
                words.append(eval(line[:-1]))
        return words
    
    def set_quantization_limits(self):
        def limit(a):
            hist, bin_edges = np.histogram(a, bins=20, density=True)
            int = np.array([np.sum((hist * np.diff(bin_edges))[:k]) for k in range(len(hist))])
            return bin_edges[np.argmax(int > 1 - Tokenizer.margin)]
        
        shift = 0
        time_set = []
        value_set = []
        temperature_set = []

        while shift + Tokenizer.window < len(self.data):
            clazz = ls.piecewise(
                value=[(_[1][0][0] + _[1][0][1]) / 2 for _ in self.data[shift: shift + Tokenizer.window]], 
                filter=Tokenizer.filter, 
                number_pieces=Tokenizer.number_pieces)
            xk, yk = clazz.knots()
            time_set.extend(np.diff(xk))
            value_set.extend(yk)
            temperature_set.extend(clazz.temperature())
            shift += Tokenizer.window
        
        Tokenizer.time_qizer =  Quantizator(limit(time_set))
        Tokenizer.value_qizer = Quantizator(limit(value_set))
        
        Tokenizer.temperature_qizer = Quantizator(limit(temperature_set))

    def get_sentence(self, data):
        """Converts a short chunk of data into tokens.

        :param data: List of tuples, each consisted of a timestamp, tuple of two 
        (ASK and BID) candles, and volume.
        :return: List of tuples, each having the token and its timestamp.

        Takes a list of tuples, each consisted of a timestamp, tuple of two 
        (ASK and BID) candles, and volume e.g: 
        `(
            1672841640.0, 
            (
                [-0.001587, -0.001267, -0.001737, -0.001437], 
                [-0.001637, -0.001317, -0.001787, -0.0014773]
            ), 
            1055900.0244
        )`.

        Produces a list of tuples, each having the token and its timestamp e.g:
        `(
            (
                19.14, 4.83e-05, -0.00727
            ), 
            1672782932.5137246
        )'
        """
        self.clazz = ls.piecewise(
            value=[(_[1][0][0] + _[1][0][1]) / 2 for _ in data], 
            filter=Tokenizer.filter, number_pieces=Tokenizer.number_pieces)
        # import pdb; pdb.set_trace()
        time_set, value_set = self.clazz.knots()
        self.time_part = [time_set[i] - time_set[i-1] for i in range(1, len(time_set))]
        time_qu = Tokenizer.time_qizer.quantize(self.time_part)
        time0 = data[0][0]
        time_qu_abs = np.cumsum(time_qu) * config.PERIOD + time0

        self.value_part = [value_set[i] for i in range(1, len(value_set))]
        self.temp_part = self.clazz.temperature()
        value_qu = Tokenizer.value_qizer.quantize(self.value_part)
        temp_qu = Tokenizer.temperature_qizer.quantize(self.temp_part)
        
        token = [((time_qu[i], temp_qu[i], value_qu[i]), time_qu_abs[i]) for i in range(len(time_qu))]
        return token
    
    def get_story(self, data, save=False, window=120):
        """Converts data into tokens.

        :param data: List of tuples, each consisted of a timestamp, tuple of two 
        (ASK and BID) candles, and volume.
        :param save: Optional, if set, the result is saved to file.
        :param window: Length of the data chunk tokenized in a single batch of
        conversion.
        :return: List of tuples, each having the token and its timestamp.

        Takes a list of tuples, each consisted of a timestamp, tuple of two 
        (ASK and BID) candles, and volume e.g: 
        `(
            1672841640.0, 
            (
                [-0.001587, -0.001267, -0.001737, -0.001437], 
                [-0.001637, -0.001317, -0.001787, -0.0014773]
            ), 
            1055900.0244
        )`.

        Produces a list of tuples, each having the token and its timestamp e.g:
        `(
            (
                19.14, 4.83e-05, -0.00727
            ), 
            1672782932.5137246
        )'
        """
        shift = 0
        token_plus = []
        data = td.DATA
        while shift + window < len(data):
                token_plus.extend(self.get_sentence(data[shift: shift + window]))
                shift += window
        if save:
            Tokenizer.save_story(token_plus)
        return token_plus

class Quantizator:
    def __init__(self, limit, level_count=8):
        self.levels = np.array([(lambda x: (2 ** x))(x) for x in range(level_count)])
        self.levels = self.levels * (limit / self.levels[-1])

    def approx(self, x):
        x_ = math.fabs(x)
        if x_ <= self.levels[0]:
            retval = self.levels[0]
            if x < 0 : retval *= -1
            return retval
        if x_ >= self.levels[-1]:
            retval = len(self.levels) - 1
            if x < 0 : retval += 8
            return retval

        for i in range(1, len(self.levels)):
            if x_ < self.levels[i]:
                mean = (self.levels[i-1] * self.levels[i]) ** .5
                if x_ <= mean: retval = self.levels[i-1]
                if x_ >= mean: retval = self.levels[i]
                if x < 0: retval *= -1
                return retval
            
    def quantize(self, x):
        return np.array([self.approx(_) for _ in x])


def test_temperature():
    window = 120
    shift = 0
    step = 100
    temperature = []

    while shift + window < len(td.VALUE):
        clazz = ls.piecewise(
            value=[(_[0] + _[1]) / 2 for _ in td.VALUE[shift: shift + window]], 
            filter=co.Savgol_filter(window=50, order=5), 
            number_pieces=10, k=1)
        temperature.extend(clazz.temperature())
        shift += step
    temperature = np.array(temperature)
    hist, bin_edg = np.histogram(temperature, 10)
    print('hist:\n', hist)
    print('edges:\n', bin_edg)


def test_quantization():
    tokenizer = Tokenizer(td.VALUE)
    tokenizer.set_quantization_limits()
    shift = 0
    window = 120
    time_value_temp = []
    while shift + window < len(td.VALUE):
        time_value_temp.extend(tokenizer.get_sentence(td.VALUE[shift: shift + window]))
        shift += window
    # import pdb; pdb.set_trace()
    print(set(Tokenizer.get_sentence_str(time_value_temp)))


def test():
    import test_data as td
    from test_data import set_test_data
    from tokenizer import Tokenizer

    set_test_data(
        data_count=10000, 
        moving_av=True)

    tokenizer = Tokenizer(td.DATA)
    tokenizer.set_quantization_limits() 
    data = td.DATA
    token_plus = tokenizer.get_story(data)
    
    for i in range(3):
        print(f'token[{i}]:\n {token_plus[i]}')
    
    import pdb; pdb.set_trace()
    token = [_[0] for _ in token_plus]

    def plot_token(token_plus):

        begin_index = next((i for i, t in enumerate(data) if t[0] > token_plus[0][1]), -1)
        end_index = next((i for i, t in enumerate(data) if t[0] > token_plus[0][1]), -1)
        nonlocal previous
        previous = None
        def decode_token(tp, x):
            if (diff := (previous[0][0] - tp[0][0])) == 0:
                return tp[1]
            
            previous = tp
            return (tp[0][2] - previous[0][2]) / diff * x 
           
        first = True
        time_tp = []
        value_tp = []
        time = data[begin_index][1]
        for tp in token_plus:
            if first:
                previous = tp
                first = False
            
            while time < tp[0]:
                pass


    plot_token(token_plus)



    current = ((2.3935439343241414, 4.836509748507084e-05, -0.007276000929348797), 1672781783.612636)
    for tp in token_plus:
        previous = current
        current = tp







    # len(time_temp_value)
    # with open(
    #     path.join(DATA_STORE, f'whole_story_{Tokenizer.window}_{Tokenizer.number_pieces}.txt'), 'w') as f:
    #     for word in whole_story:
    #         f.write(f'{str(word)}\n')

def main():
    test()
    # td.set_test_data(data_count=20000, moving_av=True)
    # test_quantization()
    # test_nn_input()
    # test_temperature()

if __name__ == "__main__":
    main()