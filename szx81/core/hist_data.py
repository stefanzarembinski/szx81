import time
import datetime
import os
from os import path
import re
import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
import core.core as co

def save_data(data):
    file = path.join(co.DATA_STORE, f'hist_data.pkl')
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to file {file}')

def get_data_from_file(set_levels=True):  
    try:   
        with open(path.join(co.DATA_STORE, f'hist_data.pkl'), 'rb') as f:
            data = pickle.load(f)
    except:
        return None
    return data

class HistData:
    """
        Reads data from all files in a `test_data`, specified with the combined 
        definitions in the `SETUP` and CONFIG maps.
        Loads the data to a list accessible with a generator defined in the Class.
    """
    def __init__(self, data_count=200, start_time=None, moving_av=True):
        _warmup_time = co.config.WARMUP_TIME if moving_av else 0           

        if data_count is not None:
            data_count += _warmup_time
        if start_time is not None:
            start_time -= _warmup_time * co.config.PERIOD * 60
        self.data_count = data_count
        self.start_time = start_time
        
        self.iter_count = 0
        self.data_dir = path.join(co.TEST_DATA_DIR, co.TEST_DATA)
        ask_files = []
        bid_files = []
        
        for data_file in os.listdir(self.data_dir):
            rex = co.config.FILE_FORMAT
            pair, direction, date_from, date_till = re.match(rex, data_file).groups()
            if pair != co.config.NAME:
                raise RuntimeError('Wrong forex currency pair!')
            strptime = datetime.datetime.strptime(date_from, co.config.FILENAME_TIMESTRING)
            timestamp_from = time.mktime(strptime.timetuple())
            strptime = datetime.datetime.strptime(date_till, co.config.FILENAME_TIMESTRING)
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
        # Subtract moving average:
        self.data = {}

        ma = co.MovingAverage()
        _warmup_time -= 1
        previous = None
        for dat in data_map:
            timestamp = dat[0]
            value = dat[1]
            if value == previous:
                # import pdb; pdb.set_trace()
                continue
            # import pdb; pdb.set_trace()
            previous = value
            mean_value = 0
            if moving_av:
                ma.add((value[0][0] + value[1][0]) / 2)
                _warmup_time -= 1
                if _warmup_time > 0:
                    continue
                mean_value = ma.ma()
            volume = dat[1][2]
            self.data[timestamp] = (timestamp, 
                              ([val - mean_value  for val in value[0]], 
                               [val - mean_value for val in value[1]]), volume )
        if len(self.data) < 10:
            raise RuntimeError('The resulting data count is very small!')

    def read_data_files(self, direction):
        start_time = self.start_time
        data_count = self.data_count
        timestring_format = co.config.TIME_STRING

        for data_file in self.directions[direction]:
            
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
                    
                    if not timestamp % co.config.PERIOD == 0:
                        raise RuntimeError('Wrong data period!')

                    _values = []
                    for i in range(1, len(values) - 1):
                        _values.append(float(values[i]))
                    
                    volume = float(values[-1])

                    if timestamp in self.data_map:
                        # import pdb; pdb.set_trace()
                        self.data_map[timestamp][direction] = _values
                    else:
                        entry = [None] * 3
                        entry[direction] = _values
                        entry[2] = volume
                        self.data_map[timestamp] = entry
          
DICT_DATA = None
DATA = None
VALUE = None
TIMESTAMP = None

def ForexProvider():
    """Generator providing forex data.
    """
    return ((timestamp, value) for timestamp, value in DICT_DATA.items())

def set_hist_data(data_count=3000, start_time=None, moving_av=True, force_save=False, 
                  verbose=False):
    global DICT_DATA
    global DATA
    global VALUE
    global TIMESTAMP

    DICT_DATA = None
    if data_count is None:
        if not force_save:
            DICT_DATA = get_data_from_file()

    if DICT_DATA is None:
        DICT_DATA = HistData(data_count=data_count, start_time=start_time, moving_av=moving_av).data
        if data_count is None:
            save_data(DICT_DATA)

    DATA = list(DICT_DATA.values())
    VALUE = np.array([(values[1][0][0], values[1][1][0]) for values in DATA])
    TIMESTAMP = np.array([_[0] for _ in DATA])
    if verbose:
        print(f'Test data size (flats are duducted) is {len(DATA)}')
        print(f'Test data start time is {time.strftime("%Y:%m:%d %H:%M", time.gmtime(DATA[0][0]))}')
        print(f'Test data end time is   {time.strftime("%Y:%m:%d %H:%M", time.gmtime(DATA[-1][0]))}')
        print(f'Subtracting moving avarage: {moving_av}')

    return DATA

def plot(count=None):
    plot_values = DATA[:count] if count is not None else PendingDeprecationWarning
    x = []
    ask = []
    bid = []
    diff = []
    _x = 0
    deltax = config_all.PERIOD / 60
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

def save_to_file():
    set_hist_data(
        data_count=None, 
        moving_av=True,
        force_save=True
    )    

def main():
    # save_to_file()

    set_hist_data(data_count=None, moving_av=True)
    plot(12000)

if __name__ == "__main__":
    main()
