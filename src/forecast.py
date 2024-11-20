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
import hist_data as td

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

def main():
    td.set_test_data(data_count=20000, moving_av=True)

if __name__ == "__main__":
    main()