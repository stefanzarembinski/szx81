import datetime
import sys
import os.path as path
import math
import matplotlib.pyplot as plt
import numpy as np

DIRNAME = path.dirname(__file__)
sys.path.append(path.join(DIRNAME, '..'))
from core import *
import test_data as td

if td.TEST_DATA_OBJ is None:
    td.set_test_data(
    data_size=5000, 
    start_time=datetime.datetime(2023, 3, 21, 12, 24).timestamp(),
    moving_av=False
    )
from test_data import *


CURRENT_RANGE = 11 * 60
FUTURE_RANGE = 1 * 60
Y_RANGE = [-2e-3, 2e-3]
SAVE_SIZE = 2 ** 7

class LogPlot:
    def __init__(self, offset=1e-5):
        self.offset = offset
        
    def convert(self, x):
        if type(x) is (int or float):
            return \
                math.log10(max(0, x) + self.offset) if x > 0 else \
                math.log10(-min(0, x) + self.offset)
        return [math.log10(max(0, _) + self.offset) if _ > 0 else \
                -math.log10(-min(0, _) + self.offset) for _ in x]

class ExpQuench:
    def __init__(self, tau=66):
        self.tau = tau

    def quench(self, t):
        return np.array([math.exp(-math.fabs((_/self.tau))) for _ in t])

class GaussQuench:
    def __init__(self, tau=66):
        self.tau = tau

    def quench(self, t):
        return np.array([math.exp(-(_/self.tau) ** 2) for _ in t])
    
class FiltrK:
    def __init__(self, tau=66, exp=0.5):
        self.tau = tau
        self.exp = exp

    def quench(self, t):
        return np.array([math.exp(-math.fabs((_/self.tau)) ** self.exp) for _ in t])

# ZANIK = ExpQuench(tau=60)
ZANIK = GaussQuench(tau=60)
# ZANIK = FiltrK(tau=66, exp=0.5)

CURRENT_FIG_DIR = path.normpath(path.join(DIRNAME,'../../obrazki_kubusia/current'))
FUTURE_FIG_DIR = path.normpath(path.join(DIRNAME,'../../obrazki_kubusia/future'))
TIMESTRING = '%y-%m-%d_%H-%M'
# import pdb; pdb.set_trace()

# FILTER = lambda value: Savgol_filter(window=50, order=5).filter(value)
FILTER = Savgol_filter(window=50, order=5)

class Plot:
    def __init__(self, plotter=plt, axis_off=False):
        self.plotter = plotter
        self.axis_off = axis_off

    def file_name(self, dir):
        date_time = datetime.datetime.fromtimestamp(self.timestamp)       
        data_string = date_time.strftime(TIMESTRING)
        return path.join(dir, data_string + '.png')
    
    def off(self):
        if self.axis_off:
            self.plotter.axis('off')
        return self.axis_off
    
    def plot(self):
        pass
    
    def save(self, dir, show=False, verbose=False):
        self.plotter.figure(figsize=(4, 4), dpi=SAVE_SIZE)
        self.plot()
        self.plotter.savefig(self.file_name(dir))
        if verbose:
            print(f'SAVED: {self.file_name(dir)}')
        if not show:
            plt.clf()
            plt.close()

class PlotCurrent(Plot):
    def __init__(self, time_start=0, plotter=plt, axis_off=False):
        super().__init__(plotter=plotter, axis_off=axis_off)

        value = VALUE[time_start: time_start + CURRENT_RANGE]
        filtered_value = FILTER.filter(value)
        filtered_value = filtered_value - filtered_value[-1]
        self.time_count = np.array([i for i in range(CURRENT_RANGE)], dtype='float64')
        self.time_count = self.time_count - self.time_count[-1]
        self.zanik = filtered_value * ZANIK.quench(self.time_count)
                        
        self.timestamp = TIMESTAMP[time_start + CURRENT_RANGE]
        
    def plot(self):
        self.plotter.plot(self.time_count, self.zanik, color='black', linewidth=0.0)
        self.plotter.fill_between(self.time_count, self.zanik, where=((self.zanik < 0)), color='blue')
        self.plotter.fill_between(self.time_count, self.zanik, where=((self.zanik > 0)), color='red')
        self.off()
        self.plotter.axis([-CURRENT_RANGE, 0, Y_RANGE[0], Y_RANGE[1]])

class PlotFuture(Plot):
    def __init__(self, time_start=0, plotter=plt, axis_off=False):
        super().__init__(plotter=plotter, axis_off=axis_off)

        value = VALUE[time_start + CURRENT_RANGE: time_start + CURRENT_RANGE + FUTURE_RANGE]
        self.value = value - value[0]
        self.time_count = np.array([i for i in range(FUTURE_RANGE)], dtype='float64')

        self.timestamp = TIMESTAMP[time_start]        

    def plot(self):
        self.plotter.plot(self.time_count, self.value, color='black', linewidth=0.5)
        self.off()
        self.plotter.axis([0, CURRENT_RANGE, Y_RANGE[0], Y_RANGE[1]])

class PlotBoth(Plot):
    def __init__(self, time_start=0, plotter=plt, axis_off=False):
        super().__init__(plotter=plotter, axis_off=axis_off)       
        value = VALUE[time_start: time_start + CURRENT_RANGE]

        filtered_value = FILTER.filter(value)
        filtered_value = filtered_value - filtered_value[-1]
        self.time_current = np.array([i for i in range(CURRENT_RANGE)], dtype='float64')
        self.time_current = self.time_current - self.time_current[-1]
        self.zanik = filtered_value * ZANIK.quench(self.time_current)
        
        value = VALUE[time_start: time_start + CURRENT_RANGE + FUTURE_RANGE]
        time = np.array([i for i in range(len(value))], dtype='float64')
        self.value = value - value[CURRENT_RANGE]
        self.time = time - time[CURRENT_RANGE]
        
        self.timestamp = TIMESTAMP[time_start + CURRENT_RANGE]
        
    def plot(self):
        self.plotter.plot(self.time, self.value, color='black', linewidth=0.5)

        self.plotter.plot(self.time_current, self.zanik, color='black', linewidth=0.0)
        self.plotter.fill_between(self.time_current, self.zanik, where=((self.zanik < 0)), color='blue')
        self.plotter.fill_between(self.time_current, self.zanik, where=((self.zanik > 0)), color='red')
        self.plotter.axvline(x=2, color='black', linestyle='--', linewidth=1)
        self.off()
        self.plotter.axis([-CURRENT_RANGE, FUTURE_RANGE, Y_RANGE[0], Y_RANGE[1]])


def dwa_obrazki(time_start=0, axis_off=False):
    fig, (current, both) = plt.subplots(1, 2, figsize=(10, 5))

    plot = PlotCurrent(time_start, current, axis_off=axis_off)
    plot.plot()
    print(plot.file_name(CURRENT_FIG_DIR))

    plot = PlotBoth(time_start, both, axis_off=axis_off)
    plot.plot()
    print(plot.file_name(FUTURE_FIG_DIR))

    plt.show()

def save(time_start=0):
    plot = PlotCurrent(time_start, plt, axis_off=True)
    plot.save(CURRENT_FIG_DIR)
    
    plot = PlotBoth(time_start, plt, axis_off=True)
    plot.save(FUTURE_FIG_DIR)
    plt.cla()
    plt.close()

def remove_all():
    import os, shutil
    def from_folder(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)        

    from_folder(CURRENT_FIG_DIR)
    from_folder(FUTURE_FIG_DIR)

def test_log_plot(time_start=0):
    lp = LogPlot()
    value = VALUE[time_start: time_start + CURRENT_RANGE]
    value = lp.convert(value)
    time_count = np.array([i for i in range(len(value))], dtype='float64')
    plt.plot(time_count, LogPlot().convert(value), label='values', color='green', linewidth=0.2)
    plt.legend()
    plt.show()

def main():
    # test_log_plot()
    dwa_obrazki(time_start=20, axis_off=False)
    # save()

if __name__ == "__main__":
    main()

