import datetime
import math
import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.signal as signal
import core as co
import piecewise_fit as ls

import test_data as td
if td.DATA is None:
    td.set_test_data(
    data_size=20000, 
    start_time=datetime.datetime(2023, 3, 21, 12, 24).timestamp(),
        moving_av=True
    )

class Forecast:
    PIP = 1e-4 # XTB min spread is 0.5 pip for STANDARD account. Market spread min. is 0.1 pip
    def __init__(self, value, threshold=None, plotall=False):

        self.value = value
        self.threshold = co.config.CONFIG['forecast_threshold'] \
            if threshold is None else threshold
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
  
        def set_direction(direction, opposite):
            # import pdb; pdb.set_trace()
            if direction is None:
                return
            self.direction = direction
            self.index_min = np.argmin(direction())
            self.min = direction()[self.index_min] 
            self.index_max = np.argmax(direction())
            self.max = direction()[self.index_max]
            self.end = direction()[-1]
            self.begin_price = opposite()[0]
            self.min_end_price = self.begin_price + self.threshold \
                if direction == self.ask else -self.threshold      
            difference = self.min_end_price - self.begin_price
            self.min_profit = difference if direction == self.ask else -difference

        # import pdb; pdb.set_trace()

        # (buy price ask now - low) - (sell price in bid future - high) > -(min profit)
#         print(f'''
# threshold + spread[0]: {(self.threshold + spread[0]) / Forecast.PIP:.1f}              

# Buy (ask) low, sell (bid) high:
# ask[0]: {self.ask()[0] / Forecast.PIP:.1f}
# max bid: {max(self.bid()) / Forecast.PIP:.1f}
# diff: {(max(self.bid()) - self.ask()[0]) / Forecast.PIP:.1f}

# Sell (bid) high, buy (ask) low:
# bid[0]: {self.bid()[0] / Forecast.PIP:.1f}
# min ask: {min(self.ask()) / Forecast.PIP:.1f}
# diff: {(self.bid()[0] - min(self.ask())) / Forecast.PIP:.1f}
# ''')              
        if max(self.bid()) - self.ask()[0] > self.threshold + spread[0]:
            # bid (buy-sell)
            set_direction(self.bid, self.ask)

        # (sell price - bid now - high) - (buy price ask in future - low) > (min profit)
        elif self.bid()[0] - min(self.ask()) > self.threshold + spread[0]:
            # ask (sell-buy)
            set_direction(self.ask, self.bid)
            
    def __str__(self):
        str = 'forecast:\n'
        if self.direction is None:
            str += f'direction: none'
            return str
                
        str += f'min: {self.min:.2e}\n'
        str += f'max: {self.max:.2e}\n'
        direction =  'bid (buy-sell)'
        if self.direction == self.ask:
            direction = 'ask: sell-buy'
        str += f'direction: {direction}\n'
        str += f'min profit: {self.min_profit:.2e}\n'
        return str
    
    def plot(self):
        cndl_count = np.array([i for i in range(-len(self.value) + 1, 1)], dtype='float64')
        plt.plot(cndl_count, self.ask() / Forecast.PIP, label='ask')
        plt.plot(cndl_count, self.bid() / Forecast.PIP, label='bid')
        if self.direction == self.bid:
            plt.scatter([cndl_count[0]], self.ask()[0] / Forecast.PIP, label='buy level')
            plt.hlines((self.ask()[0] + self.threshold) / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], label='min sell level')
            if not self.plotall:
                plt.axis([cndl_count[0], cndl_count[-1], 
                    -1e-4, 3 * self.threshold / Forecast.PIP])

        elif self.direction == self.ask: 
            plt.scatter([cndl_count[0]], self.bid()[0] / Forecast.PIP, label='sell level')
            plt.hlines((self.bid()[0] - self.threshold) / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], label='min buy level')
            if not self.plotall:
                plt.axis([cndl_count[0], cndl_count[-1], 
                    -3 * self.threshold / Forecast.PIP, 1e-4])
        
        # plt .scatter([cndl_count[self.index_max]], self.max, label='buy level')
        
        plt.legend()
        plt.show()


FORECAST_WINDOW = co.config.CONFIG['forecast_window']
FORECAST_WINDOW = 30
FORECAST_THRESHOLD = 2e-4

def main():
    shift = 1600
    forecasts = []
    forecast = Forecast(
        td.VALUE[shift: FORECAST_WINDOW + shift], 
        threshold=FORECAST_THRESHOLD, plotall=True)
    print(forecast)
    forecast.plot()


    # while shift + FORECAST_WINDOW < len(td.VALUE):
    #     forecasts.append(Forecast(td.VALUE[shift: FORECAST_WINDOW + shift]))


if __name__ == "__main__":
    main()