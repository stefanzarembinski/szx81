import sys
from os import path
import numbers
import time
import datetime
import numpy as np
import scipy.signal as signal
import setup

SETUP = setup.CONFIG
TEST_DATA_DIR = path.join(path.dirname(__file__), '../', SETUP['forex'])
sys.path.append(TEST_DATA_DIR)
import config # type: ignore

TEST_DATA = 'test_data'
WARMUP_TIME = config.MA_WINDOW_SIZE # min

class MovingAverage:
    
    def __init__(self, window_size=config.MA_WINDOW_SIZE):
        self.window_size = window_size
        self.window_data = []
        self.count = 0

    def add(self, value):
        if self.count >= self.window_size:
            self.window_data.pop(0)
        else:
            self.count += 1
        self.window_data.append(value)

    def ma(self):
        return sum(self.window_data) / self.count

class Savgol_filter:
    def __init__(self, window=50, order=2):
        self.window = window
        self.order = order

    def filter(self, values):
        return signal.savgol_filter(values, self.window, self.order)
