import sys
import os
from os import path
import numbers
import time
import datetime
import csv
import matplotlib.pyplot as plt
import setup

SETUP = setup.CONFIG
TEST_DATA_DIR = path.join(path.dirname(__file__), '../', SETUP['forex'])
sys.path.append(TEST_DATA_DIR)
import config # type: ignore

TEST_DATA = 'test_data'
WARMUP_TIME = 120 # min

class MovingAverage:
    def __init__(self, window_size=config.CONFIG['ma_window_size']):
        self.window_size = window_size
        self.window_data = []
        self.count = 0

    def add(self, value):
        self.window_data.append(value)
        if self.count > self.window_size:
            self.window_data.pop[0]
        else:
            self.count += 1

    def ma(self):
        return sum(self.window_data) / self.count
    
class TestData:
    """
Reads data from all files in a `test_data`, specified with the combined definitions in
the `SETUP` and CONFIG maps.
Loads the data to a list accesible with a generator defined in the Class.
    """
    def __init__(self):
        timestring_format = config.CONFIG['timestring']
        data = {} # `using dict` to avoid duplicates
        self.count = 0
        data_dir = path.join(TEST_DATA_DIR, TEST_DATA)
        for data_file in os.listdir(data_dir):
            with open(path.join(data_dir, data_file), newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                next(spamreader)
                for line in spamreader:
                    values = line[2].split(',')
                    timestring = line[0] + ' ' + line[1] + ' ' + values[0] 
                    strptime = datetime.datetime.strptime(timestring, timestring_format)
                    timestamp = time.mktime(strptime.timetuple())
                    if not timestamp % config.CONFIG['period'] == 0:
                        raise RuntimeError('Wrong data period!')

                    _values = []
                    for i in range(1, len(values)):
                        _values.append(float(values[i]))
                    data[timestamp] = _values

        # Sort data because datafile order can be not sequential:
        data = sorted(data.items())

        # Subtract mooving avarage:
        start_time = None
        ma = MovingAverage()
        self.data = []
        for dat in data:
            timestamp = dat[0]
            if start_time is None:
                start_time = timestamp
            ma.add(dat[1][0])

            if (timestamp - start_time) / 60 < WARMUP_TIME:
                continue
            mean_value = ma.ma()
            values = [val - mean_value for val in dat[1]]
            self.data.append((timestamp, values))

    
    def reset_generator(self, count=0):
        if count != 0:
            if isinstance(count, numbers.Number):
                timestamp = int(count)
                for index in range(len(self.data) - 1, -1, -1):
                    if self.data[index][0] <= timestamp: break
                count = index
            
        self.count = count

    def generator(self):
        if self.count < len(self.data):
            retval = self.data[self.count]
            self.count += 1 
            yield retval
        yield None
        
    def plot(self, count=None):
        plot_values = self.data[:count] if count is not None else self.data
        x = []
        y = []
        _x = 0
        deltax = config.CONFIG['period'] / 60
        for _xy in plot_values:
            x.append(_x)
            _x += deltax
            y.append(_xy[1][1])

        plt.plot(x, y, label='forex', color='green')
        plt.xlabel('time [min]')
        plt.ylabel('open value')
        plt.legend()
        plt.show()

def main():
    test_data = TestData()
    test_data.plot()