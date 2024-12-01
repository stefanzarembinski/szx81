import sys
from os import path
import importlib
import scipy.signal as signal
from IPython.display import HTML, display, Markdown, Latex

import config_all

def _(text):
    return display(Markdown(text))
HTML("""
<style type="text/css">
body{
    width:600px /* or you can also set 90% or 900px*/
    column-width: 600px
}
p {
    font-size:10px;
}
</style>
""")

def jupiter_dir():
    # https://stackoverflow.com/questions/52119454/how-to-obtain-jupyter-notebooks-path
    import os
    return os.path.abspath('')

# import setup

SETUP = config_all.CONFIG
TEST_DATA_DIR = path.join(path.dirname(__file__), '../', SETUP['forex'])
sys.path.append(TEST_DATA_DIR)
import config # type: ignore
importlib.reload(config_all)

TEST_DATA = 'test_data'
WARMUP_TIME = config.MA_WINDOW_SIZE # min
DATA_STORE = path.join(TEST_DATA_DIR, config.DATA_STORE)

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
