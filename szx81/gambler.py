import sys
import matplotlib.pyplot as plt

import hist_data as hd
import forecast as fo

PIP = 1e-4

class Gambler:
    """
    """
    gamblers = []
    profits = []

    @classmethod
    def new_instance(cls, forex_prediction, min_profit, 
                 risk_factor, strategy_class=None):
        """Creates a ``Gambler`` class object and includes it into the working
            batch. It proceeds till the end of the transaction when it is killed.
        """
        inst = cls(forex_prediction, min_profit, risk_factor, strategy_class)
        cls.gamblers.append(inst)
        return inst
    
    @classmethod
    def clear_gamblers(cls, force=False):
        for g in cls.gamblers:
            if g.is_ready or force and g.direction is not None:
                cls.profits.append((g.timestamp, g.direction, len(g.times), g.profit))
                cls.gamblers.remove(g)

    @classmethod
    def step_gamblers(cls, forex_prediction):
        """
        Parameters
        ----------
        forex_prediction : Generator providing tuple consisting both 
            new forex and oracle prediction
        """
        cls.clear_gamblers()
        for g in Gambler.gamblers:
            g.gamble(forex_prediction)

    class DefaultStrategy:
        """Envelope for strategy definition of ``Gambler`` objects.
        Must have a ``strategy`` method.
        """
        def __init__(gambler):
            pass

        def strategy(self, _):
            """Strategy definition
            Parameters
            ----------
            _ : Reference to the Gambler object.

            Returns
            -------
            continue : If set, continue transaction, finish otherwise.
            """
            if (_.profit >= _.min_profit) and _.max_profit is None:
                _.max_profit = -sys.float_info.max

            if _.max_profit is not None:
                _.max_profit = max(_.max_profit, _.profit)
                if _.profit < _.max_profit * _.risk_factor:
                    return False
            return True

    def __init__(self, 
                 forex_prediction, min_profit=3*PIP , 
                 risk_factor=0.7, strategy_class=None):
        """
        Parameters
        ----------
        forex_prediction : Generator of recentest forex data.
        strategy_class: Class having a ``strategy`` method.
        """
        self.forex_prediction = forex_prediction
        self.min_profit = min_profit
        self.risk_factor = risk_factor
        if strategy_class is None: strategy_class = Gambler.DefaultStrategy
        self.strategy_object = strategy_class()
        
        self.buy_price = None
        self.sell_price = None
        self.max_profit = None
        self.ask = None
        self.bid = None
        self.direction = None
        self.profit = None

        self.times = []
        self.ask_prices = []
        self.bid_prices = []
        self.profits = []
        self.time = 0
        self.timestamp = None
        self.end_of_file = False
        self.is_ready = False

    def __get_profit(self, forex_prediction=None):
        if forex_prediction is not None:
            (timestamp, (timestamp, (ask, bid), volume)), \
            (advice, trans_time, panic, max_panic_time) = forex_prediction
        else:
            (timestamp, (timestamp, (ask, bid), volume)), \
                (advice, trans_time, panic, max_panic_time) = self.forex_prediction()      

        # set ``self.direction`` once only
        if (advice is None) and (self.direction is None):
            return None
        if self.direction is None: 
            self.direction = advice
            self.timestamp = timestamp

        self.ask = ask[0]
        self.bid = bid[0]
        
        if self.direction == fo.Forecast.advices[fo.Forecast.IS_ASK]: # sell-buy
            if self.sell_price is None:
                self.sell_price = self.bid
            profit = self.sell_price - self.ask
        elif self.direction == fo.Forecast.advices[fo.Forecast.IS_BID]: # buy-sell
            if self.buy_price is None:
                self.buy_price = self.ask
            profit = self.bid - self.buy_price
        else:
            profit = None
        
        if profit is not None:
            self.times.append(self.time)
            self.time += 1
            self.ask_prices.append(self.ask)
            self.bid_prices.append(self.bid)
            self.profits.append(profit)
        
        self.profit = profit

    def gamble(self, forex_prediction=None):
        while True:
            try:
                self.__get_profit(forex_prediction)
            except Exception as ex:
                self.end_of_file = True
                print('END OF DATA')
                break
                
            # gambling strategy:
            if self.direction is not None:
                if not self.strategy_object.strategy(self):
                    self.is_ready = True
                    break

            if forex_prediction is not None:
                break

    def plot(self):
        _, bid_ask_pl = plt.subplots()
        bid_ask_pl.plot(self.times, self.bid_prices, color='blue', label='bid')
        bid_ask_pl.plot(self.times, self.ask_prices, color='red', label='ask')

        profit_pl = bid_ask_pl.twinx()
        profit_pl.hlines(self.min_profit, self.times[0], 
            self.times[-1], color='black', linestyle='--', label='min. profit')
        profit_pl.plot(self.times, self.profits, color='green', label='profit')

        profit_pl.set_ylabel('profit, min. profit, panic', color='green')
        bid_ask_pl.set_ylabel('bid, ask', color='black')
        
        profit_pl.legend(loc='upper left')
        bid_ask_pl.legend(loc='lower right')
        plt.show()

def prices(data):
    timestamp, (timestamp, (ask, bid), volume)  = data
    return ask[0], bid[0]

def gambler(forex_prediction, min_profit=3*PIP, risk_factor=0.7, plot=False, strategy_class=None):
    g = Gambler(
        forex_prediction=forex_prediction, min_profit=min_profit, risk_factor=risk_factor,
            strategy_class=strategy_class)
    g.gamble()

    if plot:
        print(f'advice: {g.direction}, profit: {g.profit:.1e}' \
                + f', min. profit: {g.min_profit:.1e}')
        g.plot()            
    return g

def run(forex_prediction, min_profit=3*fo.Forecast.PIP, 
                 risk_factor=0.8, strategy_class=None, 
                 time_step=5, verbose=False):
    """Runs many ``Gambler`` instances, ``time_step`` one after the previous.
    Parameters
    ----------
    forex_prediction : Generator of recentest forex data.
    strategy_class: Class having a ``strategy`` method.

    Returns
    -------
    profits : List of tuples ``(timestamp, direction, len(times), profit)``
    """
    timestamp_prev = -5 * 60
    Gambler.gamblers.clear()
    Gambler.profits.clear

    while True:
        try:
            fp = forex_prediction()
        except:
            print('END OF DATA')
            Gambler.clear_gamblers(force=True)
            break

        if fp[0][0] - timestamp_prev > time_step * 60:
            timestamp_prev = fp[0][0]
            Gambler.new_instance(fp, min_profit, risk_factor, strategy_class)
        Gambler.step_gamblers(fp)

    if verbose:
        total = 0
        
        for _ in Gambler.profits:
            total += _[3]
        print(f'profit: {total:.1e}, trans. count: {len(Gambler.profits)}')
    
    return Gambler.profits

def test_run():
    hd.set_hist_data(data_count=None)
    oracle = fo.Oracle(hd.ForexProvider())    
    
    run(forex_prediction=oracle.prediction, verbose=True)        

def test():
    hd.set_hist_data(data_count=None)
    oracle = fo.Oracle(hd.ForexProvider())

    g = gambler(forex_prediction=oracle.prediction, plot=True)
    print(f'advice: {g.direction}, profit: {g.profit:.1e}' \
                    + f', min. profit: {g.min_profit:.1e}')
 
def main():
    # test()
    test_run()

if __name__ == "__main__":
    main()  