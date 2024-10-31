import time
import datetime
import os
from os import path
import numpy as np
import csv
import matplotlib.pyplot as plt
import core as co

class TestData:
    """
Reads data from all files in a `test_data`, specified with the combined definitions in
the `SETUP` and CONFIG maps.
Loads the data to a list accesible with a generator defined in the Class.
    """
    def __init__(self, data_count=200, start_time=None, moving_av=True):

        _warmup_time = co.WARMUP_TIME if moving_av else 0
        if data_count is not None:
            data_count += _warmup_time
        if start_time is not None:
            start_time -= _warmup_time * co.PERIOD * 60
        
        timestring_format = co.TIMESTRING
        data = {} # `using dict` to avoid duplicates
        self.iter_count = 0
        data_dir = path.join(co.TEST_DATA_DIR, co.TEST_DATA)
        for data_file in os.listdir(data_dir):
            with open(path.join(data_dir, data_file), newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                next(spamreader)
                for line in spamreader:
                    line = line[0] + ' ' + line[1]
                    values = line.split(',')
                    timestring = values[0]
                    strptime = datetime.datetime.strptime(timestring, timestring_format)
                    timestamp = time.mktime(strptime.timetuple())
                    
                    if start_time is not None:
                        if timestamp > start_time:
                            start_time = None
                            data = {}
                    
                    if data_count is not None and start_time is None:
                        if data_count <= 0:
                            break
                        data_count -= 1
                    
                    if not timestamp % co.PERIOD == 0:
                        raise RuntimeError('Wrong data period!')

                    _values = []
                    for i in range(1, len(values) - 1):
                        _values.append(float(values[i]))
                    data[timestamp] = _values

        # Sort data because datafile order can be not sequential:
        data = sorted(data.items())

        # Subtract mooving avarage:
        self.data = []
        ma = co.MovingAverage()
        _warmup_time -= 1
        for dat in data:
            timestamp = dat[0]
            _warmup_time -= 1
            if _warmup_time > 0:
                continue
            
            if moving_av:
                ma.add(dat[1][0])
                mean_value = ma.ma()
                values = [val - mean_value for val in dat[1]]
            else:
                values = [val for val in dat[1]]

            self.data.append((timestamp, values))
        pass
            
    def reset_generator(self, count=0):
        if count != 0:
            if isinstance(count, (int, float)):
                timestamp = int(count)
                for index in range(len(self.data) - 1, -1, -1):
                    if self.data[index][0] <= timestamp: break
                count = index
            
        self.iter_count = count

    def generator(self):
        if self.iter_count < len(self.data):
            retval = self.data[self.iter_count]
            self.iter_count += 1 
            yield retval
        yield None

DATA = None
VALUE = None
TIMESTAMP = None

def set_test_data(data_size=3000, start_time=None, moving_av=True):
    global DATA
    global VALUE
    global TIMESTAMP

    DATA = TestData(data_count=data_size, start_time=start_time, moving_av=moving_av).data
    VALUE = np.array([values[1][0] for values in DATA])
    TIMESTAMP = np.array([values[0] for values in DATA])
    print(f'Test data size is {len(DATA)}')
    print(f'Test data start time is {time.strftime("%Y:%m:%d %H:%M", time.gmtime(DATA[0][0]))}')
    print(f'Test data end time is   {time.strftime("%Y:%m:%d %H:%M", time.gmtime(DATA[-1][0]))}')
    print(f'Subtracting moving avarage: {moving_av}')
    
def plot(count=None):
    plot_values = DATA[:count] if count is not None else PendingDeprecationWarning
    x = []
    y = []
    _x = 0
    deltax = co.PERIOD / 60
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
    set_test_data(
    data_size=5000, 
    start_time=datetime.datetime(2023, 3, 21, 11, 25).timestamp(),
    moving_av=False
    )
    plot(12000)

if __name__ == "__main__":
    main()
