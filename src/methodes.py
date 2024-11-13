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

class Forecast:
    PIP = 1e-4 # XTB min spread is 0.5 pip for STANDARD account. Market spread min. is 0.1 pip
    def __init__(self, value, threshold=None, plotall=False):

        self.value = value
        self.threshold = co.config.CONFIG['forecast_threshold'] \
            if threshold is None else threshold
        panic_level = 1e-4
        
        self.plotall = plotall
        
        self.mean = (np.array([_[0] for _ in value]) + np.array([_[1] for _ in value])) / 2
        self.mean = self.mean - self.mean[0]
        spread = (np.array([_[0] for _ in value]) - np.array([_[1] for _ in value]))
        ask = (self.mean + spread / 2)
        bid = (self.mean - spread / 2)
        self.ask = lambda: ask # ask is the price a seller is willing to accept
        self.bid = lambda: bid # bid is the price a buyer is willing to pay

        self.min = min(self.mean)
        self.max = max(self.mean)
        self.direction = None

        '''
threshold + spread[0]: {(self.threshold + spread[0]) / Forecast.PIP:.1f}              

Buy (ask) low, sell (bid) high:
ask[0]: {self.ask()[0] / Forecast.PIP:.1f}
max bid: {max(self.bid()) / Forecast.PIP:.1f}
diff: {(max(self.bid()) - self.ask()[0]) / Forecast.PIP:.1f}

:
bid[0]: {self.bid()[0] / Forecast.PIP:.1f}
min ask: {min(self.ask()) / Forecast.PIP:.1f}
diff: {(self.bid()[0] - min(self.ask())) / Forecast.PIP:.1f}
''' 
        def set_direction(direction, opposite):
            if direction is None:
                return
            self.direction = direction
            self.index_min = np.argmin(direction())
            self.min = direction()[self.index_min] 
            self.index_max = np.argmax(direction())
            self.max = direction()[self.index_max]
            self.end = direction()[-1]
            self.begin_price = opposite()[0]
            
            self.panic = 0
            if direction == self.bid:
                self.min_end_price = self.begin_price - self.threshold                
                self.panic_level = opposite()[0] - panic_level
                m = min((direction() < self.panic_level) * direction()) 
                if m < 0:
                    self.panic = math.fabs(m - self.panic_level)
            elif direction == self.ask:
                self.min_end_price = self.begin_price + self.threshold
                self.panic_level = opposite()[0] + panic_level
                m = max((direction() > self.panic_level) * direction())
                if m > 0:
                    self.panic = math.fabs( - self.panic_level)
            
            difference = self.min_end_price - self.begin_price
            self.min_profit = - difference

        # import pdb; pdb.set_trace()
        # (buy price ask now - low) - (sell price in bid future - high) > -(min profit)
        if max(self.bid()) - self.ask()[0] > self.threshold + spread[0]:
            # bid (buy-sell)
            set_direction(self.bid, self.ask)

        # (sell price - bid now - high) - (buy price ask in future - low) > (min profit)
        elif self.bid()[0] - min(self.ask()) > self.threshold + spread[0]:
            # ask (sell-buy)
            set_direction(self.ask, self.bid)

    def forecast(self):
        if self.direction is None:
            return (0, None)
        return (
            -1 if self.direction == self.ask else 1,
            self.panic
            )
            
    def __str__(self):
        str = 'forecast:\n'
        if self.direction is None:
            str += f'direction: none'
            return str
        direction =  'bid (buy-sell)'
        if self.direction == self.ask:
            direction = 'ask: sell-buy'
        str += f'direction: {direction}\n'
        str += f'min profit [PIP]: {self.min_profit / Forecast.PIP:.2f}\n'
        str += f'panic value [PIP]: {self.panic / Forecast.PIP:.2f}\n'
        return str
    
    def plot(self):
        cndl_count = np.array([i for i in range(len(self.value))], dtype='float64')
        plt.plot(cndl_count, self.ask() / Forecast.PIP, label='ask')
        plt.plot(cndl_count, self.bid() / Forecast.PIP, label='bid')
        if self.direction == self.bid:
            threshold_level = self.ask()[0] + self.threshold
            plt.scatter([cndl_count[0]], self.ask()[0] / Forecast.PIP, label='buy point')
            sell = np.argmax(self.bid() > threshold_level)
            plt.scatter([cndl_count[sell]], self.bid()[sell] / Forecast.PIP, label='sell point')

            plt.hlines(self.panic_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], color='red', label='panic level',
                       linestyle='--')
            plt.hlines(threshold_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], label='min sell level')
            
            if not self.plotall:
                plt.axis([cndl_count[0], cndl_count[-1], 
                    -1e-4 / Forecast.PIP, 2 * self.threshold / Forecast.PIP])

        elif self.direction == self.ask:
            threshold_level = self.bid()[0] - self.threshold
            plt.scatter([cndl_count[0]], self.bid()[0] / Forecast.PIP, label='sell point')
            buy = np.argmax(self.ask() < threshold_level)
            plt.scatter([cndl_count[buy]], self.ask()[buy] / Forecast.PIP, label='buy point')
            
            plt.hlines(self.panic_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], color='red', label='panic level',
                       linestyle='--')
            plt.hlines(threshold_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], label='min buy level')
            
            if not self.plotall:
                plt.axis([cndl_count[0], cndl_count[-1], 
                    -2 * self.threshold / Forecast.PIP, 1e-4 / Forecast.PIP])
        
        # plt .scatter([cndl_count[self.index_max]], self.max, label='buy level')
        
        plt.legend()
        plt.show()

class Tokenizer:
    number_pieces = 10
    window = 120
    margin = 0.05
    time_qizer = None
    value_qizer = None
    temperature_qizer = None
    filter = co.Savgol_filter(window=50, order=5)
    none_word = '0000'

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

    def get_sentence_str(sequence, words=None):
        retval = [f'{oct(time)[2:]}{oct(temp)[2:]}{oct(value)[2:].rjust(2, "0")}' 
                for (time, temp, value) in sequence]
        if words is not None:
            retval = [_ if _ in words else Tokenizer.none_word for _ in retval]
        return retval
    
    def get_sentence_bytes(sequence):
        bytes = []
        for (time, temp, value) in sequence:
            bytes.extend([time + 8 * temp, value])
        return bytes
    def __init__(self, value_limits):
        self.value_limits = value_limits
        self.clazz = None
        self.time_part = None
        self.value_part = None
        self.temp_part = None
    
    def set_quantization_limits(self):
        def limit(a):
            hist, bin_edges = np.histogram(a, bins=20, density=True)
            int = np.array([np.sum((hist * np.diff(bin_edges))[:k]) for k in range(len(hist))])
            return bin_edges[np.argmax(int > 1 - Tokenizer.margin)]
        
        shift = 0
        time_set = []
        value_set = []
        temperature_set = []

        while shift + Tokenizer.window < len(self.value_limits):
            clazz = ls.piecewise(
                value=[(_[0] + _[1]) / 2 for _ in self.value_limits[shift: shift + Tokenizer.window]], 
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

    def get_sentence(self, value, save=False):
        self.clazz = ls.piecewise(
            value=[(_[0] + _[1]) / 2 for _ in value], 
            filter=Tokenizer.filter, number_pieces=Tokenizer.number_pieces)
        time_set, value_set = self.clazz.knots()
        self.time_part = [time_set[i] - time_set[i-1] for i in range(1, len(time_set))]
        time_qu = Tokenizer.time_qizer.quantize(self.time_part)
        self.value_part = [(value_set[i] + value_set[i-1]) / 2 for i in range(1, len(value_set))]
        self.temp_part = self.clazz.temperature()
        value_qu = Tokenizer.value_qizer.quantize(self.value_part)
        temp_qu = Tokenizer.temperature_qizer.quantize(self.temp_part)
        # import pdb; pdb.set_trace()
        retval = [(time_qu[i], temp_qu[i], value_qu[i]) for i in range(len(time_qu))]
        if save:
            words = collections.Counter(Tokenizer.get_sentence_str(retval)).most_common()
            Tokenizer.save_words(words)
        return retval

class Quantizator:
    def __init__(self, limit, level_count=8):
        self.levels = np.array([(lambda x: (2 ** x))(x) for x in range(level_count)])
        self.limit = limit

    def approx(self, x):
        x_ = math.fabs(x) * self.levels[-1] / self.limit
        if x_ <= self.levels[0]:
            return 0
        if x_ >= self.levels[-1]:
            retval = len(self.levels) - 1
            if x < 0 : retval += 8
            return retval

        for i in range(1, len(self.levels)):
            if x_ < self.levels[i]:
                mean = (self.levels[i-1] * self.levels[i]) ** .5
                if x_ <= mean: retval = i-1
                if x_ >= mean: retval = i
                if x < 0: retval += 8
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

def test_forecast():
    forecast_window = config.forecast_window
    forecast_window = 30
    forecast_threshold = 2e-4 + 1e-4 # 1 pip for spread    

    shift = 1600
    forecast = Forecast(
        td.VALUE[shift: forecast_window + shift], 
        threshold=forecast_threshold, plotall=False)
    print(forecast)
    forecast.plot() 

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

def main():
    td.set_test_data(data_count=20000, moving_av=True)
    test_quantization()
    # test_nn_input()
    # test_temperature()

if __name__ == "__main__":
    main()