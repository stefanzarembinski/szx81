import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit

from core import *

np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})

test_data = TestData()


class Savgol_filter:
    def __init__(self, window=50, order=2):
        self.window = window
        self.order = order

    def filter(self, values):
        return signal.savgol_filter(values, self.window, self.order)


data = test_data.data[:1200]
x_data = [i for i in range(len(data))]
y_data = [values[1][0] for values in data]
limits = (0, 300)
shift = 0
data = data[limits[0] + shift: limits[1] + shift]
filter = Savgol_filter(window=40, order=2)

time = np.array([i + shift for i in range(len(data) + shift)], dtype='float64')
value = [values[1][0] for values in data]
value = filter.filter(value)
time = time * 1e-5


class LinPiswise:
    def __init__(self, x, y, number_pieces=10):
        self.x = x if isinstance(x, np.ndarray) else np.array(x, dtype='float64') 
        self.y = y if isinstance(y, np.ndarray) else np.array(y, dtype='float64')
        self.n = number_pieces
        
    def piecewise(self, x, *params):
        print(np.array(params))
        xk = np.array(params[:self.n])
        xk = np.append(xk, x[-1])
        xk = np.insert(xk, 0, x[0])
        yk = np.array(params[self.n:])

        funclist = np.array([], dtype='void')
        condlist = []
        
        def l(x):
            # import pdb; pdb.set_trace()
            for k in range(len(xk)): 
                if (min(x) <= xk[k]) & (xk[k] <= max(x)):
                    break
            if(k == len(xk) - 1):
                k -= 1
            # print(f'k: {k}; x: {x}')
            return (yk[k] + ((yk[k+1] - yk[k]) / (xk[k+1] - xk[k])) * (x - xk[k]))

        for k in range(self.n + 1):
            func = lambda x: l(x)
            funclist = np.append(funclist, func)
            cond = (xk[k] <= x) & (x <= xk[k+1])
            condlist.append(cond)
        
        condlist = np.array(condlist)
        # import pdb; pdb.set_trace()
        return np.piecewise(x, condlist, funclist)

    def param_0(self):
        p = []
        x0 = self.x[0]
        x_1 = self.x[-1]
        sep = (x_1 - x0) / (self.n + 2)

        xk = [x0]
        ik = [0]

        for i in range(1, len(self.x)):
            if len(xk) == self.n + 1:
                break
            if xk[-1] + sep < self.x[i]:
                ik.append(i)
                xk.append(0)

        if len(xk) != self.n + 1:
            raise Exception(f'Parameter 0 failure! Too many pieces ({self.n})?')
        
        ik.append(len(self.x) - 1)
        xk.append(x_1)
        param = xk[1:-1]

        yk = []
        for i in ik:
            yk.append(self.y[i])
            param.append(self.y[i])
        return np.array(param)

    def plot(self, show_plot=False):
        params = self.param_0()
        xk = params[:self.n]
        xk = np.append(xk, self.x[-1])
        xk = np.insert(xk, 0, self.x[0])
        yk = params[self.n:]
        plt.plot(xk, yk, color='black', label='parameters 0')
        plt.scatter(self.x, self.piecewise(self.x, *self.param_0()), color='blue', label='piecewise 0')
        if show_plot:
            plt.legend()
            plt.show()


class PiecewiseLinear:
    def __init__(self, x, y, func_class=LinPiswise, number_pieces=5):
        self.x = x if isinstance(x, np.ndarray) else np.array(x, dtype='float64')
        self.y = y if isinstance(y, np.ndarray) else np.array(y, dtype='float64')
        self.n = number_pieces
        self.func = func_class(x, y, number_pieces)

    def run(self):
        return curve_fit(self.func.piecewise, self.x, self.y, p0=self.func.param_0())

# lin_piecewise = LinPiswise(time, value, number_pieces=5)
# lin_piecewise.plot(True)

pl = PiecewiseLinear(time, value, number_pieces=20)
p, e = pl.run()
print(f'parameters calculated:\n{np.array(p)}')
print(f'parameters at startup:\n{pl.func.param_0()}')

plt.plot(time, value, color='green', label='data')
plt.plot(time, pl.func.piecewise(time, *p), color='blue', label='piecewise')
pl.func.plot()

plt.legend()
plt.show()
