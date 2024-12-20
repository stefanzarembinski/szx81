import sys
from os import path
import importlib
import scipy.signal as signal
from IPython.display import HTML, display, Markdown, Latex
import textwrap

import logging
logging.basicConfig(format="{levelname}:{funcName}: {message}", style="{")
# logging.basicConfig(format="{levelname}:{name}:{message}", style="{")

import config_all

def wrap_text(text, width=80):
    wrapped_text = ''
    for element in textwrap.TextWrapper(
        width=width,
        replace_whitespace=False
        ).wrap(text=text):
        wrapped_text += element + '<br>'
    return wrapped_text

def _(text, width=80):
    text = wrap_text(text, width)
    return display(Markdown(text))

def _nw(text):
    return display(Markdown(text))

# HTML("""
# <style type="text/css">
# body{
#     width:600px /* or you can also set 90% or 900px*/
#     column-width: 600px
# }
# p {
#     font-size:10px;
# }
# </style>
# """)

def jupiter_dir():
    # https://stackoverflow.com/questions/52119454/how-to-obtain-jupyter-notebooks-path
    import os
    return os.path.abspath('')

TEST_DATA_DIR = None
MA_WINDOW_SIZE = None
DATA_STORE = None
TEST_DATA = 'test_data'
DATA_DIR_NAME = None
config = None

def setup(data_dir_name):
    global TEST_DATA_DIR
    global MA_WINDOW_SIZE
    global DATA_STORE
    global config

    TEST_DATA_DIR = path.join(
        path.dirname(__file__), '../../', data_dir_name)
    sys.path.append(TEST_DATA_DIR)
    import config as _config # type: ignore
    importlib.reload(_config)
    config = _config

    MA_WINDOW_SIZE = config.MA_WINDOW_SIZE
    DATA_STORE = path.join(TEST_DATA_DIR, config.DATA_STORE)

setup(config_all.DATA_DIR_NAME)

class MovingAverage:
    
    def __init__(self, window_size=MA_WINDOW_SIZE):
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
