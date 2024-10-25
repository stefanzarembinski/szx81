
import time
import datetime
import csv
import matplotlib.pyplot as plt

def roundLog(x, toString=False):
    n = 0o7
    values = [1 / n * i for i in range(0, n)]
    l = math.log10(math.fabs(x))
    c = math.floor(l)
    m = l - c
    m = values[min(range(len(values)), key = lambda i: abs(values[i]-m))]
    xx = 10 ** (c + m)
    xx = math.copysign(xx, x)
    return f"{xx:0.1e}" if toString else xx

class MovingAverage:
    def __init__(self, window_size=24*60*60):
        self.window_size = window_size
    

TEST_EURUSD = './EURUSD/test_data/EURUSD_Candlestick_1_M_BID_05.09.2024-05.10.2024.csv'
TIMESTRING = '%d.%m.%Y %H:%M:%S.000 GMT%z'


import math
import numpy as np
FIT_ORDER = 2
START_LEN = FIT_ORDER + 1
DATA_PERIOD = 60
THRESHOLD = 10.0

class Parser:
    def __init__(self, data_source, testData=None):
        self.testData = testData          
        self.data = data_source
        self.last_values = next(self.data())
        if self.last_values is None:
            print('Data source is empty!')
            return 
        self.tokens = []
        while True:
            if self.last_values is None:
                break
            self.token()

    def token(self):
        token_time = self.last_values[0]
        times = [(self.last_values[0] - self.last_values[0]) / DATA_PERIOD]
        values = [self.last_values[1]]
        for i in range(START_LEN):
            self.last_values = next(self.data())
            if self.last_values is None:
                break
            times.append((self.last_values[0] - token_time) / DATA_PERIOD)
            values.append(self.last_values[1])
        
        while True:
            self.last_values = next(self.data())
            if self.last_values is None:
                if len(times) > START_LEN:
                    coefficients = np.polyfit(times, values, FIT_ORDER)
                    polynomial = np.poly1d(coefficients)
                    self.tokens.append((token_time, len(times), polynomial))
                return
            times.append((self.last_values[0] - token_time) / DATA_PERIOD)
            values.append(self.last_values[1])
            coefficients, residuals, _, _, _ = np.polyfit(times, values, FIT_ORDER, full=True)
            polynomial = np.poly1d(coefficients)
            error = (polynomial(times[-1]) - self.last_values[1]) ** 2 \
                / (residuals[0] / len(times))
            if error > THRESHOLD:
                times = times[:-2] 
                values = values[:-2]
                coefficients = np.polyfit(times, values, FIT_ORDER)
                # import pdb; pdb.set_trace()
                for i in range(0, FIT_ORDER):
                    coefficients[i] = roundLog(coefficients[i])
                polynomial = np.poly1d(coefficients)
                self.tokens.append((token_time, len(times), polynomial))
                break

    def plot(self, count=None):
        timeOffset = None 
        if self.testData is not None:
            plot_values = self.testData.data[:count] if count is not None else self.testData.data
            timeOffset = plot_values[0][0]
            plt.plot(
                [int((value[0] - timeOffset) / 60) for value in plot_values], 
                [value[1] for value in plot_values], 
                label='forex', color='green')
        
        colors = ('red', 'blue')
        color = True
        first = 2
        for entry in self.tokens:
            if not timeOffset: timeOffset = entry[0]
            color = not color
            x = [t + (entry[0] - timeOffset) / 60 for t in range(entry[1])]
            if count is not None:
                count -= len(x)
                if count < 0:
                    break
            y = entry[2]([t for t in range(entry[1])])
            # import pdb; pdb.set_trace()
            if first:
                first -= 1
                plt.plot(x, y, label='tokeny', color=colors[int(color)], linewidth=5, alpha=0.5)
            plt.plot(x, y, color=colors[int(color)], linewidth=5, alpha=0.5)
           
        
        plt.xlabel('time [min]')
        plt.ylabel('open')
        plt.legend()
        plt.show()


    
