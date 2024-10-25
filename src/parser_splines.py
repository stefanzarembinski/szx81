import math
import numpy as np
import matplotlib.pyplot as plt
from core import *

FIT_ORDER = 2
START_LEN = FIT_ORDER + 1
DATA_PERIOD = 60
THRESHOLD = 10.0

import scipy
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.interpolate import make_interp_spline
from scipy.interpolate import splrep

test_data = TestData()
data = test_data.data[:1200]
x = [i for i in range(len(data))]
y = [values[1][0] for values in data]

s = 0.00005
tck = splrep(x, y, s=s, k=2)
poly = scipy.interpolate.PPoly.from_spline(tck)

import pdb; pdb.set_trace()

plt.plot(x, BSpline(*tck)(x), '-', label=f's={s}')
plt.plot(x, y, '.', alpha=0.1)
plt.legend()
plt.show()



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

