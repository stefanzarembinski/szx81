import collections
from os import path
import math
import numpy as np
import pickle
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
    level_count = 8
    time_levels = None
    value_levels = None
    temperature_levels = None
    filter = co.Savgol_filter(window=50, order=5)
    none_word = '0000'

    def __init__(self):
        pass
    
    def get_statistics(data, bins=100, force=False):
        """Provides statistics of the token attributes: `time`, `value`, `temperature`.

        Parameters
        ----------
        data : List of tuples, each consisted of a timestamp, tuple of two 
            (ASK and BID) candles, and volume.
        bins : The number of histogram bins.
        force : If set, do not reload statistics from file.

        Returns
        -------- 
        time_stat : tuple ``(hist, bin_edges)`` for 'time' statistics.
        value_stat : tuple ``(hist, bin_edges)`` for 'value' statistics.
        temperature_stat : tuple ``(hist, bin_edges)`` for 'temperature' statistics.
        """        
        
        file = path.join(
            co.DATA_STORE, 
            f'statistics_{Tokenizer.window}_{Tokenizer.number_pieces}.pkl')
        
        time_set = []
        value_set = []
        temperature_set = []  
        # import pdb; pdb.set_trace()
        if not force and path.exists(file):
            with open(file, 'rb') as f: 
                time_set, value_set, temperature_set = pickle.load(f) 
               
        if (len(time_set) == 0) or (len(value_set) == 0) or (len(temperature_set) == 0):
            shift = 0
            while shift + Tokenizer.window < len(data):
                clazz = ls.piecewise(
                    value=[(_[1][0][0] + _[1][0][1]) / 2 for _ in data[shift: shift + Tokenizer.window]], 
                    filter=Tokenizer.filter, 
                    number_pieces=Tokenizer.number_pieces)
                xk, yk = clazz.knots()
                time_set.extend(np.diff(xk))
                value_set.extend(yk)
                temperature_set.extend(clazz.temperature())
                shift += Tokenizer.window

            with open(file, 'wb') as f: 
                pickle.dump((time_set, value_set, temperature_set), f) 
            # with open(file, 'rb') as f: time_set_, value_set_, temperature_set_ = pickle.load(f)

        return (
            (np.histogram(time_set, bins=bins, density=True)),
            (np.histogram(value_set, bins=bins, density=True)),
            (np.histogram(temperature_set, bins=bins, density=True)),
        ) 
    
    def round_sig(x, sig=3):
        x_ = math.fabs(x)
        retval = round(x_, sig-int(math.floor(math.log10(abs(x_))))-1)
        if x < 0: retval *= -1
        return retval

    def set_quantization_levels(time_stat, value_stat, temperature_stat):
        def histogram(hist, bin_edges, level_count):
            hist_cum = hist.cumsum()
            hist_cum = hist_cum / hist_cum[-1]
            
            level_dist = 1 / (level_count + 1)
            levels = []
            for i in range(level_count):
                index = (np.abs(hist_cum - (i + 1) * level_dist).argmin())
                levels.append(bin_edges[index])

            return [Tokenizer.round_sig(_) for _ in levels]
        
        time_levels = histogram(*time_stat, Tokenizer.level_count)
        value_levels = histogram(*value_stat, 2 * Tokenizer.level_count)
        temperature_levels = histogram(*temperature_stat, Tokenizer.level_count)

        Tokenizer.time_levels = time_levels
        Tokenizer.value_levels = value_levels
        Tokenizer.temperature_levels = temperature_levels
 
    def quantize(x, levels):
        """Returns given array quantized according to given levels.

        Parameters
        ----------
        x : Array of numbers to be quantized.
        levels : Array of quantization values.

        Returns
        -------
        x_quant : Quantized input array. 
        """
        def do(x):
            if x <= levels[0]:
                return levels[0]
            if x >= levels[-1]:
                return levels[-1]

            for i in range(1, len(levels)):
                if x < levels[i]:
                    mean = (levels[i-1] + levels[i]) * .5
                    if x <= mean: return levels[i-1]
                    return levels[i]
        return np.array([do(_) for _ in x])

    def get_sentence(data):
        """Converts a short chunk of data into tokens.
        
        Parameters
        ----------
        data : List of tuples, each consisted of a timestamp, tuple of two 
        (ASK and BID) candles, and volume.
        
        Returns
        -------
        token : List of tokens.

        Notes
        -----
        Quantum levels has to be established with :py:func:`Tokenizer.set_quantization_levels' 
        function.

        Takes a list of tuples, each consisted of a timestamp, tuple of two 
        (ASK and BID) candles, and volume e.g:

        ``` 
            (
                1672841640.0, 
                (
                    [-0.001587, -0.001267, -0.001737, -0.001437], 
                    [-0.001637, -0.001317, -0.001787, -0.0014773]
                ), 
                1055900.0244
            )
        ```

        Produces a list of tuples, each having the token and its timestamp e.g:

        ```
            (
                (
                    19.14, 4.83e-05, -0.00727
                ), 
                1672782932.5137246
            )
        ```
        
        """
        
        clazz = ls.piecewise(
            value=[(_[1][0][0] + _[1][0][1]) / 2 for _ in data], 
            filter=Tokenizer.filter, number_pieces=Tokenizer.number_pieces)
        
        time_set, value_set = clazz.knots()
        time_part = [time_set[i] - time_set[i-1] for i in range(1, len(time_set))]
        time_qu = Tokenizer.quantize(time_part, Tokenizer.time_levels)
        
        value_part = [value_set[i] for i in range(1, len(value_set))]
        value_qu = Tokenizer.quantize(value_part, Tokenizer.value_levels)
        
        # import pdb; pdb.set_trace()        
        temp_part = clazz.temperature()
        temp_qu = Tokenizer.quantize(temp_part, Tokenizer.temperature_levels)
        
        return [(round(time_qu[i], 1), value_qu[i], temp_qu[i]) 
                                                    for i in range(len(time_qu))]
    
    def get_whole_story(data, window=None, force_save=False):
        """Converts data into tokens.

        Parameters
        ----------
        data : List of tuples, each consisted of a timestamp, tuple of two 
            (ASK and BID) candles, and volume.
            :param save: Optional, if set, the result is saved to file.
        window : Length of the data chunk tokenized in a single batch of
        conversion.

        Returns
        -------
        token : List of tokens.

        Notes
        -----
        Takes a list of tuples, each consisted of a timestamp, tuple of two 
        (ASK and BID) candles, and volume e.g:
        
        ```
            (
                1672841640.0, 
                (
                    [-0.001587, -0.001267, -0.001737, -0.001437], 
                    [-0.001637, -0.001317, -0.001787, -0.0014773]
                ), 
                1055900.0244
            )
        ```

        Produces a list of tokens e.g:

        ```
            (19.14, 4.83e-05, -0.00727)
        ```

        """

        file = path.join(
            co.DATA_STORE, 
            f'whole_story_{Tokenizer.window}_{Tokenizer.number_pieces}.pkl')        
        token = []

        if not force_save and path.exists(file):
            with open(file, 'rb') as f: 
                token = pickle.load(f) 

        if len(token) == 0:
            if window is None:
                window = Tokenizer.window

            shift = 0
            while shift + window < len(data):
                    token.extend(Tokenizer.get_sentence(data[shift: shift + window]))
                    shift += window

            with open(file, 'wb') as f: 
                pickle.dump(token, f)
        
        return token

    def get_words_used(whole_story):
        word_count = collections.Counter(whole_story).most_common()
        return np.array([_[0] for _ in word_count])
        
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
    tokenizer = Tokenizer(td.Data)
    Tokenizer.set_quantization_levels(Tokenizer.get_statistics(td.DATA, bins=100))

    shift = 0
    window = 120
    time_value_temp = []
    while shift + window < len(td.DATA):
        time_value_temp.extend(tokenizer.get_sentence(td.DATA[shift: shift + window]))
        shift += window
    # import pdb; pdb.set_trace()
    print(set(Tokenizer.get_sentence_str(time_value_temp)))


def test():
    import test_data as td
    from test_data import set_test_data
    from tokenizer import Tokenizer

    set_test_data(
    data_count=None, 
    moving_av=True)

    DATA = td.DATA

    Tokenizer.set_quantization_levels(*Tokenizer.get_statistics(DATA, bins=100))

    shift = 1000 
    window = 120
    token = Tokenizer.get_sentence(DATA[shift: shift + window])

    token = Tokenizer.get_whole_story(DATA)
    import pdb; pdb.set_trace()
    pass

def main():
    test()
    # td.set_test_data(data_count=20000, moving_av=True)
    # test_quantization()
    # test_nn_input()
    # test_temperature()

if __name__ == "__main__":
    main()