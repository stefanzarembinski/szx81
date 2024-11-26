import sys
import matplotlib.pyplot as plt

import hist_data as hd
import forecast as fo

PIP = 1e-4

class GumblerSimple:
    """
    """
    gamblers = []
    total = 0

    @classmethod
    def get_instance(cls, oracle, min_profit=3*fo.Forecast.PIP, 
                 risk_factor=0.7):
        inst = cls(oracle, min_profit, risk_factor)
        cls.gamblers.append(inst)
        return inst
    
    @classmethod
    def clear_gamblers(cls):
        for g in cls.gamblers:
            if g.profit is not None:
                cls.total += g.profit 
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
        forex_prediction  = next(forex_prediction)
        for g in GumblerSimple.gamblers:
            g.gamble(forex_prediction)

    def __init__(self, 
                 forex_prediction, min_profit=3*PIP , 
                 risk_factor=0.7):
        """
        Parameters
        ----------
        forex_prediction : Generator of recentmost forex data.
        oracle: Generator of recentmost predictios.
        """
        self.forex_prediction = forex_prediction
        self.min_profit = min_profit
        self.risk_factor = risk_factor
        
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

        while True:
            self.__get_profit()
            if self.direction is not None:
                break

    def __get_profit(self): 
        (timestamp, (timestamp, (ask, bid), volume)), \
            (advice, trans_time, panic, max_panic_time) = self.forex_prediction()
        
        # set ``self.direction`` once only
        if (advice is None) and (self.direction is None):
            return
        if self.direction is None: self.direction = advice

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

        return profit 

    def gamble(self):
        while True:
            profit = self.__get_profit()

            # gambling trategy:
            if (profit >= self.min_profit) and self.max_profit is None:
                self.max_profit = -sys.float_info.max

            if self.max_profit is not None:
                self.max_profit = max(self.max_profit, profit)
                if profit < self.max_profit * self.risk_factor:
                    break

        self.profit = profit              

    def get_profit(self):
        """To check whether process is finished: if return is not None.
        Then the profit can be booked and the object distroyed.
        """        
        return self.profit

    def plot(self):
        _, bidask_pl = plt.subplots()
        bidask_pl.plot(self.times, self.bid_prices, color='blue', label='bid')
        bidask_pl.plot(self.times, self.ask_prices, color='red', label='ask')

        profit_pl = bidask_pl.twinx()
        profit_pl.hlines(self.min_profit, self.times[0], 
            self.times[-1], color='black', linestyle='--', label='min. profit')
        profit_pl.plot(self.times, self.profits, color='green', label='profit')

        profit_pl.set_ylabel('profit, min. profit, panic', color='green')
        bidask_pl.set_ylabel('bid, ask', color='black')
        
        profit_pl.legend(loc='upper left')
        bidask_pl.legend(loc='lower right')
        plt.show()

def prices(data):
    timestamp, (timestamp, (ask, bid), volume)  = data
    return ask[0], bid[0]

def gambler(forex_prediction, min_profit=3*PIP, risk_factor=0.7, plot=False):
    try:
        g = GumblerSimple(
            forex_prediction=forex_prediction, min_profit=min_profit, risk_factor=risk_factor)
        g.gamble()
        g.get_profit()
    except Exception as ex:
        print('END OF DATA')
        return

    if plot:
        print(f'advice: {g.direction}, profit: {g.profit:.1e}' \
                + f', min. profit: {g.min_profit:.1e}')
        g.plot()            
    return g

def test():
    hd.set_hist_data(data_count=None)
    oracle = fo.Oracle(hd.ForexProvider())

    g = gambler(forex_prediction=oracle.prediction, plot=True)
    print(f'advice: {g.direction}, profit: {g.profit:.1e}' \
                    + f', min. profit: {g.min_profit:.1e}')
 
def main():
    test()

if __name__ == "__main__":
    main()  