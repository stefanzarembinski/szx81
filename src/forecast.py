import collections
from os import path
import math
import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import core as co
from core import config
import piecewise_fit as ls
import hist_data as hd

class Forecast:
    """Given historical data, determines current trade opportunities.
    Parameters:
    -----------
    data : Historical data chunk investigated.

        ```
        (
            1672871700.0, 
            (
                [0.00166904166666626, 0.00166904166666626, 0.001589041666666402, 0.001629041666666442], 
                [0.001609041666666311, 0.001609041666666311, 0.001529041666666453, 0.0015790416666663365]
            ), 
            56240.0017
        )   
        ```
    threshold: The minimal profit expeched.
    plotall: If set, if result is plotted, than wider plot showing more details.
     
    Attributes:
    -----------
    panic : The maximum value of possible loss if panicked. Negative value means no panic.?
    """
    IS_ASK = True
    IS_BID = False
    advices = {IS_ASK: 'sell-buy', IS_BID: 'buy-sell'}
    PIP = 1e-4 # XTB min spread is 0.5 pip for STANDARD account. Market spread min. is 0.1 pip
    FILE = path.join(
            co.DATA_STORE,
            f'trend_{round(config.FORECAST_THRESHOLD / PIP)}_' \
        + f'{config.FORECAST_WINDOW}_.pkl')
    
    @classmethod
    def dump(cls, prediction, file=FILE):
        with open(file, 'wb') as f:
            pickle.dump(prediction, f)

    @classmethod
    def load(cls, file=FILE):
        with open(file, 'rb') as f:
            prediction = pickle.load(f)
        return prediction
    
    def __init__(self, data, threshold=None):
        self.data = data
        self.threshold = config.FORECAST_THRESHOLD \
            if threshold is None else threshold
        self.panic = None
        self.advice = None
        self.trans_time = None
        self.max_panic_time = None

        self.file = path.join(
            co.DATA_STORE, 
            f'trend_{round(self.threshold / Forecast.PIP)}_{len(data)}_.pkl')
        
        self.panic_threshold = config.PANIC_THRESHOLD
        self.mean = (np.array([_[1][0][0] for _ in data]) + np.array([_[1][1][0] for _ in data])) / 2
        self.mean = self.mean - self.mean[0]
        spread = (np.array([_[1][0][0] for _ in data]) - np.array([_[1][1][0] for _ in data]))
        ask = (self.mean + spread / 2)
        bid = (self.mean - spread / 2)
        # import pdb; pdb.set_trace()
        self.ask = lambda: ask # ask is the price a seller is willing to accept
        self.bid = lambda: bid # bid is the price a buyer is willing to pay
        self.directions = {Forecast.IS_ASK: self.ask, Forecast.IS_BID: self.bid}

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
        def set_direction(is_ask):
            """Called when trend direction is determined. It sets statistics.
            Parameters:
            -----------
            is_ask : `True` if direction is ASK, `False` if it is not.
            """
            if is_ask is None:
                return
            
            self.direction = self.directions[is_ask]
            self.opposite = self.directions[not is_ask]
            self.index_min = np.argmin(self.direction())
            self.min = self.direction()[self.index_min] 
            self.index_max = np.argmax(self.direction())
            self.max = self.direction()[self.index_max]
            self.end = self.direction()[-1]
            self.begin_price = self.opposite()[0]
            self.advice = None
            
            self.panic = None
            if not is_ask: # (buy-sell) buy low, sell high
                self.threshold_level = self.ask()[0] + self.threshold
                self.trans_time = np.argmax(self.bid() - self.threshold_level > 0)
                if self.trans_time == 0:
                    self.panic = None
                    return
                
                self.min_end_price = self.begin_price - self.threshold
                self.panic_level = self.opposite()[0] - self.panic_threshold
                self.buy_price = self.begin_price
                self.sell_price = self.direction()[self.trans_time]              
                # import pdb; pdb.set_trace()
                self.advice = Forecast.advices[is_ask]
                self.max_panic_time = np.argmax(-self.direction()[:self.trans_time])
                self.panic = -self.direction()[:self.trans_time][self.max_panic_time] + self.panic_level
                
            elif is_ask: # (sell-buy), sell high, buy low
                self.threshold_level = self.begin_price - self.threshold
                self.trans_time = np.argmax(self.ask() - self.threshold_level < 0)
                if self.trans_time == 0:
                    self.panic = None
                    return

                self.min_end_price = self.begin_price + self.threshold                      
                self.panic_level = self.opposite()[0] + self.panic_threshold
                self.buy_price = self.direction()[self.trans_time]
                self.sell_price = self.begin_price
                self.advice = Forecast.advices[is_ask]
                self.max_panic_time = np.argmax(self.direction()[:self.trans_time])
                self.panic = self.direction()[:self.trans_time][self.max_panic_time] - self.panic_level
            
            self.min_profit = self.sell_price - self.buy_price

        # import pdb; pdb.set_trace()
        # (buy price ask now - low) - (sell price in bid future - high) > -(min profit)
        if max(self.bid()) - self.ask()[0] > self.threshold + spread[0]:
            # bid (buy-sell)
            set_direction(Forecast.IS_BID)

        # (sell price - bid now - high) - (buy price ask in future - low) > (min profit)
        elif self.bid()[0] - min(self.ask()) > self.threshold + spread[0]:
            # ask (sell-buy)
            set_direction(Forecast.IS_ASK)
            

    def dump_to_file(self, prediction):
        Forecast.dump(prediction=prediction, file=self.file)

    def load_from_file(self):
        return Forecast.load(file=self.file)

    def forecast(self):
        """
        """
        if self.direction is None:
            return (0, None)
        return (
            -1 if self.direction == self.ask else 1,
            self.panic
            )
            
    def __str__(self):
        str = 'forecast:\n'
        if self.panic is None:
            str += f'direction: none\n'
            return str

        str += f'direction: {self.advice}\n'
        str += f'min profit [PIP]: {self.min_profit / Forecast.PIP:.2f}\n'
        # import pdb; pdb.set_trace()
        str += f'panic value [PIP]: {self.panic / Forecast.PIP:.2f}' \
            + (' - no panic\n' if self.panic < 0 else '\n')
        return str
    
    def plot(self, plotall=False):

        cndl_count = np.array([i for i in range(len(self.data))], dtype='float64')
        plt.plot(cndl_count, self.ask() / Forecast.PIP, label='ask')
        plt.plot(cndl_count, self.bid() / Forecast.PIP, label='bid')

        if self.direction == self.bid:
            plt.scatter([cndl_count[0]], self.ask()[0] / Forecast.PIP, 
                        marker='x', linewidths=10, label='buy point')
            plt.scatter(
                [cndl_count[self.trans_time]], 
                self.sell_price / Forecast.PIP, 
                marker='x', linewidths=10, label='sell point')
            plt.scatter(
                [cndl_count[self.max_panic_time]], 
                self.direction()[:self.trans_time][self.max_panic_time] / Forecast.PIP, 
                marker='x', linewidths=10, label='max panic point')            

            plt.hlines(self.panic_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], color='red', label='panic level',
                       linestyle='--')
            plt.hlines(self.threshold_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], label='min sell level')
            
            if not plotall:
                plt.axis([cndl_count[0], cndl_count[-1], 
                    -1e-4 / Forecast.PIP, 2 * self.threshold / Forecast.PIP])

        elif self.direction == self.ask:
            plt.scatter([cndl_count[0]], self.bid()[0] / Forecast.PIP, 
                        marker='x', linewidths=10, label='sell point')
            plt.scatter(
                [cndl_count[self.trans_time]], 
                self.buy_price / Forecast.PIP, 
                marker='x', linewidths=10, label='buy point')
            plt.scatter(
                [cndl_count[self.max_panic_time]], 
                self.direction()[:self.trans_time][self.max_panic_time] / Forecast.PIP, 
                marker='x', linewidths=10, label='max panic point')               
            
            plt.hlines(self.panic_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], color='red', label='panic level',
                       linestyle='--')
            plt.hlines(self.threshold_level / Forecast.PIP, 
                       cndl_count[0], cndl_count[-1], label='min buy level')
            
            if not plotall:
                plt.axis([cndl_count[0], cndl_count[-1], 
                    -2 * self.threshold / Forecast.PIP, 1e-4 / Forecast.PIP])
        
        plt.legend()
        plt.show()

def set_trend_predictions(data_count=None):

    hd.set_hist_data(data_count, moving_av=True)
    
    forecast_window = config.FORECAST_WINDOW
    threshold = config.FORECAST_THRESHOLD
    shift = 0
    step = 1
    prediction = {}
    
    while shift + forecast_window < len(hd.DATA):
        data = hd.DATA[shift: forecast_window + shift]
        shift += step
        forecast = Forecast(
            data, 
            threshold=threshold)
        # import pdb; pdb.set_trace()
        prediction[data[0][0]] \
            = (forecast.advice, forecast.trans_time, forecast.panic, forecast.max_panic_time)

    Forecast.dump(prediction)

def get_predictions(verbose=False):
    """
    Parameters
    ----------
        verbose: If set, print information.
    Returns
    -------
    prediction : timestamp keyed dict of tuples - 
        (advice, trans_time, panic, max_panic_time)
    sel_buy : List of ASK part. 
    buy_sell : List of BID part.
    none : No opportunity part.
    """    
    prediction = Forecast.load()
    values = prediction.values()
    sell_buy = [_ for _ in values if _[0] == Forecast.advices[Forecast.IS_ASK]]
    buy_sell = [_ for _ in values if _[0] == Forecast.advices[Forecast.IS_BID]]
    none = [_ for _ in values if _[0] is None] 
    if verbose:   
        print(f'len(sell-buy): {len(sell_buy)} ({len(sell_buy) / len(prediction) * 100:.0f}%)')
        print(f'len(buy-sell): {len(buy_sell)} ({len(buy_sell) / len(prediction) * 100:.0f}%)')
        print(f'len(none): {len(none)} ({len(none) / len(prediction) * 100:.0f}%)')
    return prediction, sell_buy, buy_sell, none

class Oracle:
    """Provides ``prediction`` function, returning prediction about the 
    future trend.
    """
    __predictions = None
    def __get_instance():
        if Oracle.__predictions is None:
            Oracle.__predictions, _, _, _ = get_predictions()
        return Oracle.__predictions
    def __init__(self):
        self.predictions_dict = Oracle.__get_instance()
    
    def predictions(self):
        """
        Returns
        -------
        predictions : timestamp keyed map of tuples 
            ``advice, trans_time, panic, max_panic_time``

        Notes
        -----
        advice : ``buy-sell`` or ``sell-buy``
        trans_time : Time point where the minimal profit is found.
        panic : ....?
        max_panic_time : Time point where the panic value is maximal.
        """
        return self.predictions_dict
    
    def prediction(self, forex):
        """Given timestamp, returns prediction about the future trend.

        parameters
        ----------
        timestamp : key value to the response.

        Returns
        -------
        prediction : tuple ``advice, trans_time, panic, max_panic_time``
        """
        return self.predictions_dict[forex[0]], forex 

def test_forecast():
    FORECAST_WINDOW = 30
    FORECAST_THRESHOLD = 2e-4 + 1e-4 # spread

    shift = 1100
    forecast = Forecast(
        hd.DATA[shift: FORECAST_WINDOW + shift], 
        threshold=FORECAST_THRESHOLD)
    
    print(forecast)
    forecast.plot(plotall=True)

def main():
    hd.set_hist_data(data_count=None)
    test_forecast()
    # set_trend_predictions()
    # prediction, sell_buy, buy_sell, none = get_predictions(verbose=True)
    # import pdb; pdb.set_trace()
    # pass

if __name__ == "__main__":
    main()