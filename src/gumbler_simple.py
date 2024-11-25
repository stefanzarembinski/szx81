import sys
import matplotlib.pyplot as plt

import hist_data as hd
import forecast as fo

class GumblerSimple:
    """
    """
    def __init__(self, 
                 direction, xtb, min_profit=3*fo.Forecast.PIP , 
                 risk_factor=0.7):
        self.direction = direction
        self.xtb = xtb
        self.min_profit = min_profit
        self.risk_factor = risk_factor
        
        self.buy_price = None
        self.sell_price = None
        self.max_profit = None

        self.times = []
        self.ask_prices = []
        self.bid_prices = []
        self.profits = []
        self.time = 0

    def __prices(self):
        timestamp, (timestamp, (ask, bid), volume)  = next(self.xtb)
        return ask[0], bid[0]
    
    def __get_profit(self, ask_price=None, bid_price=None):
        if ask_price is None:
            ask_price, bid_price = self.__prices()

        self.times.append(self.time)
        self.time += 1
        self.ask_prices.append(ask_price)
        self.bid_prices.append(bid_price)
    
        if self.direction == fo.Forecast.advices[fo.Forecast.IS_ASK]:
            if self.buy_price is None:
                self.buy_price = ask_price
            return bid_price - self.buy_price
        elif self.direction == fo.Forecast.advices[fo.Forecast.IS_BID]:
            if self.sell_price is None:
                self.sell_price = bid_price
            return  self.sell_price - ask_price
        else:
            return None        

    def gumble(self, ask_price, bid_price):
        profit = self.__get_profit(ask_price, bid_price)
        if profit is None:
            return None
        
        while True: # buy (ask) low
            profit = self.__get_profit()
            if (profit >= self.min_profit) and self.max_profit is None:
                self.max_profit = -sys.float_info.max

            if self.max_profit is not None:
                # import pdb; pdb.set_trace()
                self.max_profit = max(self.max_profit, profit)
                if profit < self.max_profit * self.risk_factor:
                    break
        return profit
    
    def plot(self):
        _, bidask_pl = plt.subplots()

        bidask_pl.plot(self.times, self.bid_prices, color='blue', label='bid')
        bidask_pl.plot(self.times, self.ask_prices, color='red', label='ask')

        profit_pl = bidask_pl.twinx()
        profit_pl.hlines(self.min_profit, self.times[0], 
            self.times[-1], color='black', linestyle='--', label='min. profit')
        profit_pl.plot(self.times, self.profits, color='green', label='profit')
        
        bidask_pl.set_ylabel('bid, ask', color='black')
        profit_pl.set_ylabel('profit, min. profit, panic', color='green')

        bidask_pl.legend()
        profit_pl.legend()
        plt.show()

def prices(data):
    timestamp, (timestamp, (ask, bid), volume)  = data
    return ask[0], bid[0]

def test():
    hd.set_hist_data(data_count=None)
    xtb = hd.XTB()
    oracle = fo.Oracle()

    while True:
        timestamp_data = next(xtb)
        if timestamp_data[0] not in oracle.predictions():
            continue
        
        ask_price, bid_price = prices(timestamp_data)
        advice, trans_time, panic, max_panic_time \
                                = oracle.prediction(timestamp_data[0])
        
        gumbler = GumblerSimple(advice, xtb)
        profit = gumbler.gumble(ask_price, bid_price)
        if profit is not None:
            print(f'profit: {profit:.1e}')
            break
 
def main():
    test()

if __name__ == "__main__":
    main()