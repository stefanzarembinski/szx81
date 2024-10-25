import numpy as np
import scipy.signal as signal
import scipy.optimize as spo
from scipy.interpolate import splrep, BSpline

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
limits = (0, 20)
shift = 0
data = data[limits[0] + shift: limits[1] + shift]
filter = Savgol_filter(window=10, order=2)

time = np.array([i + shift for i in range(len(data) + shift)], dtype='float64')
time = time * 1.e-5
value = [values[1][0] for values in data]
value = filter.filter(value)


class LinPiswise:
    def __init__(self, x, y, number_pieces=10):
        self.x = x if isinstance(x, np.ndarray) else np.array(x, dtype='float64') 
        self.y = y if isinstance(y, np.ndarray) else np.array(y, dtype='float64')
        self.n = number_pieces
        
    def piecewise(self, x, *params):
        print(np.array(params))
        # import pdb; pdb.set_trace()
        xk = np.array(params[:self.n])
        xk = np.append(xk, x[-1])
        xk = np.insert(xk, 0, x[0])
        yk = np.array(params[self.n:])

        funclist = np.array([], dtype='void')
        condlist = []
        
        def l(x):
            for k in range(len(xk)): 
                if (min(x) <= xk[k]) & (xk[k] <= max(x)):
                    break
            if(k == len(xk) - 1):
                k -= 1
            # print(f'k: {k}; x: {x}')
            # import pdb; pdb.set_trace()
            return (yk[k] + ((yk[k+1] - yk[k]) / (xk[k+1] - xk[k])) * (x - xk[k]))

        for k in range(self.n + 1):
            func = lambda x: l(x)
            funclist = np.append(funclist, func)
            cond = (xk[k] <= x) & (x <= xk[k+1])
            condlist.append(cond)
        
        condlist = np.array(condlist)
        
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
                xk.append(self.x[i])

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

    def func(self, params):
        # import pdb; pdb.set_trace()
        return sum((self.y - self.piecewise(self.x, *params)) ** 2)

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

class Splines:
    def __init__(self, x, y, number_pieces=10):
        self.x = x if isinstance(x, np.ndarray) else np.array(x, dtype='float64') 
        self.y = y if isinstance(y, np.ndarray) else np.array(y, dtype='float64')
        self.n = number_pieces

    def piecewise(self, x, *params):
        xk = np.array(params[:self.n])
        xk = np.append(xk, x[-1])
        xk = np.insert(xk, 0, x[0])
        yk = np.array(params[self.n:])
        xk, yk = [list(x) for x in zip(*sorted(zip(xk, yk), key=itemgetter(0)))]
        bspl = splrep(xk, yk, k=2, s=0)
        # import pdb; pdb.set_trace()
        spl = BSpline(*bspl)
        return spl(x)
    
    def func(self, params):
            # import pdb; pdb.set_trace()
            return sum((self.y - self.piecewise(self.x, *params)) ** 2)

    def plot(self, show_plot=False):
        params = self.param_0()
        xk = params[:self.n]
        xk = np.append(xk, self.x[-1])
        xk = np.insert(xk, 0, self.x[0])
        yk = params[self.n:]
        plt.plot(xk, yk, color='black', label='parameters 0')
        plt.plot(self.x, self.piecewise(self.x, *self.param_0()), color='blue', label='piecewise 0')
        if show_plot:
            plt.legend()
            plt.show() 
    
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
                xk.append(self.x[i])

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


class NelderMead:
    def __init__(self, func_class):
        self.func_class = func_class
    
    def run(self):
        return spo.minimize(self.func_class.func, self.func_class.param_0(),
                            method='Nelder-Mead') 
    
    
optimizer = NelderMead(Splines(time, value, 3))
p = optimizer.run()
import pdb; pdb.set_trace()

splines = Splines(time, value, 3)
splines.plot(True)

# print(p)
# print(optimizer.func_class.param_0())

# plt.scatter(time, value, color='green', label='data')
# plt.plot(time, optimizer.func_class.piecewise(time, *p), color='blue', label='piecewise')
# optimizer.func_class.plot()

# plt.legend()
# plt.show()