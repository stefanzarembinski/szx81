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
import hist_data as hd

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
            """Called when trend direction is determined. Then sets statistics.
            Parameters:
            -----------
            direction : lambda function either `self.ask` or `self.bid`.
            opposite : lambda function other then set with `direction`.
            """
            if direction is None:
                return
            self.direction = direction
            self.index_min = np.argmin(direction())
            self.min = direction()[self.index_min] 
            self.index_max = np.argmax(direction())
            self.max = direction()[self.index_max]
            self.end = direction()[-1]
            self.begin_price = opposite()[0]
            
            self.panic = None
            if direction == self.bid:
                self.min_end_price = self.begin_price - self.threshold
                self.threshold_level = self.ask()[0] + self.threshold
                self.panic_level = opposite()[0] - panic_level
                self.sell_time = np.argmax(self.bid() > self.threshold_level)               
                # import pdb; pdb.set_trace()
                self.panic = -(min(direction()[:self.sell_time]) - self.panic_level)
            elif direction == self.ask:
                self.min_end_price = self.begin_price + self.threshold
                self.threshold_level = self.begin_price - self.threshold
                self.panic_level = opposite()[0] + panic_level
                self.buy_time = np.argmax(self.ask() < self.threshold_level)
                # import pdb; pdb.set_trace()
                self.panic = max(direction()[:self.buy_time]) - self.panic_level
            
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
        # import pdb; pdb.set_trace()
        if self.panic is not None:
            str += f'panic value [PIP]: {self.panic / Forecast.PIP:.2f}' \
                + (' - no panic\n' if self.panic < 0 else '\n')
        else:
            str += 'no panic'
        return str
    
    def plot(self):
        cndl_count = np.array([i for i in range(len(self.value))], dtype='float64')
        plt.plot(cndl_count, self.ask() / Forecast.PIP, label='ask')
        plt.plot(cndl_count, self.bid() / Forecast.PIP, label='bid')
        if self.direction == self.bid:
            plt.scatter([cndl_count[0]], self.ask()[0] / Forecast.PIP, 
                        marker='x', linewidths=10, label='buy point')
            plt.scatter(
                [cndl_count[self.sell_time]], 
                self.bid()[self.sell_time] / Forecast.PIP, 
                marker='x', linewidths=10, label='sell point')

            plt.hlines(self.panic_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], color='red', label='panic level',
                       linestyle='--')
            plt.hlines(self.threshold_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], label='min sell level')
            
            if not self.plotall:
                plt.axis([cndl_count[0], cndl_count[-1], 
                    -1e-4 / Forecast.PIP, 2 * self.threshold / Forecast.PIP])

        elif self.direction == self.ask:
            plt.scatter([cndl_count[0]], self.bid()[0] / Forecast.PIP, 
                        marker='x', linewidths=10, label='sell point')
            plt.scatter(
                [cndl_count[self.buy_time]], 
                self.ask()[self.buy_time] / Forecast.PIP, 
                marker='x', linewidths=10, label='buy point')
            
            plt.hlines(self.panic_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], color='red', label='panic level',
                       linestyle='--')
            plt.hlines(self.threshold_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], label='min buy level')
            
            if not self.plotall:
                plt.axis([cndl_count[0], cndl_count[-1], 
                    -2 * self.threshold / Forecast.PIP, 1e-4 / Forecast.PIP])
        
        # plt .scatter([cndl_count[self.index_max]], self.max, label='buy level')
        
        plt.legend()
        plt.show()

def test_forecast():
    FORECAST_WINDOW = 30
    FORECAST_THRESHOLD = 2e-4 + 1e-4 # spread

    shift = 1500
    forecast = Forecast(
        hd.VALUE[shift: FORECAST_WINDOW + shift], 
        threshold=FORECAST_THRESHOLD)
    print(forecast)
    forecast.plot()

def main():
    hd.set_hist_data(data_count=None, moving_av=True)
    test_forecast()

if __name__ == "__main__":
    main()