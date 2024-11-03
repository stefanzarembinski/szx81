import time
import datetime
import os
from os import path
import re
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
        self.data_count = data_count
        self.start_time = start_time
        
        self.iter_count = 0
        self.data_dir = path.join(co.TEST_DATA_DIR, co.TEST_DATA)
        ask_files = []
        bid_files = []
        
        for data_file in os.listdir(self.data_dir):
            
            rex = co.config.CONFIG['file_format']
            pair, direction, date_from, date_till = re.match(rex, data_file).groups()
            if pair != co.SETUP['forex']:
                raise RuntimeError('Wrong forex currency pair!')
            strptime = datetime.datetime.strptime(date_from, co.config.CONFIG['filename_timestring'])
            timestamp_from = time.mktime(strptime.timetuple())
            strptime = datetime.datetime.strptime(date_till, co.config.CONFIG['filename_timestring'])
            timestamp_till = time.mktime(strptime.timetuple())
            # import pdb; pdb.set_trace()
            files = ask_files if direction == 'ASK' else bid_files
            files.append((timestamp_from, timestamp_till, data_file))
            
        ask_files = sorted(ask_files, key=lambda element: (element[0], -element[1]))
        bid_files = sorted(bid_files, key=lambda element: (element[0], -element[1]))
        self.directions = (ask_files, bid_files)
        self.data_map = {}
        self.directions = (ask_files, bid_files)
        
        self.read_data_files(0)
        self.read_data_files(1)

        # Sort data because datafile order can be not sequential:
        data_map = sorted(self.data_map.items())
        # Subtract mooving avarage:
        self.data = []
        ma = co.MovingAverage()
        _warmup_time -= 1
        previous = None
        for dat in data_map:
            timestamp = dat[0]
            value = dat[1]
            if value == previous:
                continue
            previous = value
            mean_value = 0
            if moving_av:
                ma.add((value[0][0] + value[1][0]) / 2)
                # import pdb; pdb.set_trace()
                mean_value = ma.ma()   
                _warmup_time -= 1
                if _warmup_time > 0:
                    continue

            self.data.append((timestamp, 
                              ([val - mean_value  for val in value[0]], 
                               [val - mean_value for val in value[1]])))

    def read_data_files(self, direction):
        start_time = self.start_time
        data_count = self.data_count
        timestring_format = co.config.CONFIG['timestring']

        for data_file in self.directions[direction]:
            # import pdb; pdb.set_trace()
            with open(path.join(self.data_dir, data_file[2]), newline='') as csvfile:
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
                            
                    if data_count is not None and start_time is None:
                        if data_count <= 0:
                            # import pdb; pdb.set_trace()
                            break
                        data_count -= 1
                    
                    if not timestamp % co.PERIOD == 0:
                        raise RuntimeError('Wrong data period!')

                    _values = []
                    for i in range(1, len(values) - 1):
                        _values.append(float(values[i]))

                    if timestamp in self.data_map:
                        self.data_map[timestamp][direction] = _values
                    else:
                        entry = [None] * 2
                        entry[direction] = _values
                        self.data_map[timestamp] = entry
          

DATA = None
VALUE = None
TIMESTAMP = None

def set_test_data(data_size=3000, start_time=None, moving_av=True):
    global DATA
    global VALUE
    global TIMESTAMP

    DATA = TestData(data_count=data_size, start_time=start_time, moving_av=moving_av).data
    VALUE = np.array([(values[1][0][0], values[1][1][0]) for values in DATA])
    TIMESTAMP = np.array([values[0] for values in DATA])
    print(f'Test data size is {len(DATA)}')
    print(f'Test data start time is {time.strftime("%Y:%m:%d %H:%M", time.gmtime(DATA[0][0]))}')
    print(f'Test data end time is   {time.strftime("%Y:%m:%d %H:%M", time.gmtime(DATA[-1][0]))}')
    print(f'Subtracting moving avarage: {moving_av}')
    
def plot(count=None):
    plot_values = DATA[:count] if count is not None else PendingDeprecationWarning
    x = []
    ask = []
    bid = []
    diff = []
    _x = 0
    deltax = co.PERIOD / 60
    for _xy in plot_values:
        x.append(_x)
        _x += deltax
        ask.append(_xy[1][0][0])
        bid.append(_xy[1][1][0])
        diff.append(_xy[1][0][0] - _xy[1][1][0])

    plt.plot(x, ask, label='ask')
    plt.plot(x, ask, label='bid')
    plt.xlabel('time [min]')
    plt.ylabel('open value')
    plt.legend()
    plt.show()

    plt.plot(x, diff, label='ask - bid')
    plt.xlabel('time [min]')
    plt.ylabel('open value')
    plt.legend()
    plt.show()   

def main():
    set_test_data(
    data_size=600, 
    start_time=datetime.datetime(2023, 1, 20, 13, 38).timestamp(),
    moving_av=True
    )
    plot(12000)

if __name__ == "__main__":
    main()
