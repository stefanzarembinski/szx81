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
        self.clazz = None
        self.time_part = None
        self.value_part = None
        self.temp_part = None

    def save_data(data):
        with open(path.join(
                co.DATA_STORE, 
                f'data_{Tokenizer.window}_{Tokenizer.number_pieces}.pkl'), 'wb') as f:
            pickle.dump(data, f)

    def get_data_from_file(set_levels=True):  
        try:   
            with open(
                path.join(
                    co.DATA_STORE, 
                    f'data_{Tokenizer.window}_{Tokenizer.number_pieces}.pkl'), 'rb') as f:
                data = pickle.load(f)
        except:
            return None
        return data
    
    def save_levels(time_stat, value_stat, temperature_stat, set_levels=False):
        """Derives quantization levels from data statistics and save them 
        to file.

        Parameters
        ----------
        time_stat, value_stat, temperature_stat : Data statistics as returned 
            from :py:func:`Tokenizer.get_statistics`

        set_levels : Write levels to Tokenizer definitions

        Returns
        -------
        time_level, value_level, temperature_level : Quantization levels.
        """
        time_level, value_level, temperature_level = Tokenizer.set_quantization_levels(
            time_stat, value_stat, temperature_stat, set_levels)
        
        with open(path.join(
                co.DATA_STORE, 
                f'levels_{Tokenizer.window}_{Tokenizer.number_pieces}.pkl'), 'wb') as f:
            pickle.dump((time_level, value_level, temperature_level), f)

        return time_level, value_level, temperature_level

    def get_levels_from_file(set_levels=True):
        """Loads quantization levels from file

        Parameters
        ----------
        set : Write levels to Tokenizer definitions

        Returns
        -------
        time_level, value_level, temperature_level : Quantization levels.
        """     
        with open(
            path.join(
                co.DATA_STORE, 
                f'levels_{Tokenizer.window}_{Tokenizer.number_pieces}.pkl'), 'rb') as f:
            time_levels, value_levels, temperature_levels = pickle.load(f)

        if set_levels:
            Tokenizer.time_levels = time_levels
            Tokenizer.value_levels = value_levels
            Tokenizer.temperature_levels = temperature_levels
        
        return time_levels, value_levels, temperature_levels

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
    
    def get_statistics(data, bins=20):
        """Provides statistics of the token attributes: `time`, `value`, `temperature`.

        Parameters
        ----------
        data : List of tuples, each consisted of a timestamp, tuple of two 
            (ASK and BID) candles, and volume.
        bins : The number of histogram bins.

        Returns
        -------- 
        time_stat : tuple ``(hist, bin_edges)`` for 'time' statistics.
        value_stat : tuple ``(hist, bin_edges)`` for 'value' statistics.
        temperature_stat : tuple ``(hist, bin_edges)`` for 'temperature' statistics.
        """        
        shift = 0
        time_set = []
        value_set = []
        temperature_set = []

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

    def set_quantization_levels(time_stat, value_stat, temperature_stat, set=True):
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

        if set:
            Tokenizer.time_levels = time_levels
            Tokenizer.value_levels = value_levels
            Tokenizer.temperature_levels = temperature_levels

        return time_levels, value_levels, temperature_levels
 

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

    def get_sentence(self, data):
        """Converts a short chunk of data into tokens. but only if the quantization
        limits are already  

        Parameters
        ----------
        data : List of tuples, each consisted of a timestamp, tuple of two 
        (ASK and BID) candles, and volume.
        
        Returns
        -------
        token_timestamp : List of tuples, each having the token and its timestamp.

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
        
        self.clazz = ls.piecewise(
            value=[(_[1][0][0] + _[1][0][1]) / 2 for _ in data], 
            filter=Tokenizer.filter, number_pieces=Tokenizer.number_pieces)
        
        time_set, value_set = self.clazz.knots()
        self.time_part = [time_set[i] - time_set[i-1] for i in range(1, len(time_set))]
        time_qu = Tokenizer.quantize(self.time_part, Tokenizer.time_levels)
        time0 = data[0][0]
        time_qu_abs = np.cumsum(time_qu) * config.PERIOD + time0

        self.value_part = [value_set[i] for i in range(1, len(value_set))]
        import pdb; pdb.set_trace()
        value_qu = Tokenizer.quantize(self.value_part, Tokenizer.value_levels)
        
        self.temp_part = self.clazz.temperature()
        temp_qu = Tokenizer.quantize(self.temp_part, Tokenizer.temperature_levels)
        
        token = [((round(time_qu[i], 1), value_qu[i], temp_qu[i]), 
                  round(time_qu_abs[i])) for i in range(len(time_qu))]
        return token
    
    def get_whole_story(data, save=False, window=120):
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
        token_timeshot : List of tuples, each having the token and its timestamp.

        Notes
        -----
        Takes a list of tuples, each consisted of a timestamp, tuple of two 
        (ASK and BID) candles, and volume e.g:
        
            (
                1672841640.0, 
                (
                    [-0.001587, -0.001267, -0.001737, -0.001437], 
                    [-0.001637, -0.001317, -0.001787, -0.0014773]
                ), 
                1055900.0244
            )

        Produces a list of tuples, each having the token and its timestamp e.g:

            (
                (
                    19.14, 4.83e-05, -0.00727
                ), 
                1672782932.5137246
            )

        """
        tokenizer = Tokenizer(data)
        Tokenizer.set_quantization_levels(*Tokenizer.get_statistics(data, bins=50)) 

        shift = 0
        token_plus = []
        data = td.DATA
        while shift + window < len(data):
                token_plus.extend(tokenizer.get_sentence(data[shift: shift + window]))
                shift += window
        if save:
            Tokenizer.save_story(token_plus)
        return token_plus

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
        data_count=10000, 
        moving_av=True)
    
    DATA = td.DATA
    tokenizer = Tokenizer()
    print('Tokenizer.temperature_levels: ', Tokenizer.temperature_levels)
    Tokenizer.get_levels_from_file(set_levels=True)

    shift = 1000
    window = 120
    tokenizer.get_sentence(DATA[shift: shift + window])



    # data = td.DATA
    # token_plus = Tokenizer.get_whole_story(data)
    
    # for i in range(300, 305):
    #     print(f'token[{i}]:\n {token_plus[i]}')
    
    # import pdb; pdb.set_trace()
    # token = [_[0] for _ in token_plus]

    # def plot_token(token_plus):

    #     begin_index = next((i for i, t in enumerate(data) if t[0] > token_plus[0][1]), -1)
    #     end_index = next((i for i, t in enumerate(data) if t[0] > token_plus[0][1]), -1)
    #     nonlocal previous
    #     previous = None
    #     def decode_token(tp, x):
    #         if (diff := (previous[0][0] - tp[0][0])) == 0:
    #             return tp[1]
            
    #         previous = tp
    #         return (tp[0][2] - previous[0][2]) / diff * x 
           
    #     first = True
    #     time_tp = []
    #     value_tp = []
    #     time = data[begin_index][1]
    #     for tp in token_plus:
    #         if first:
    #             previous = tp
    #             first = False
            
    #         while time < tp[0]:
    #             pass


    # plot_token(token_plus)



    # current = ((2.3935439343241414, 4.836509748507084e-05, -0.007276000929348797), 1672781783.612636)
    # for tp in token_plus:
    #     previous = current
    #     current = tp







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