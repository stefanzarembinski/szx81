import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import scipy.optimize as spo
from scipy.interpolate import splrep, BSpline

import core as co
import test_data as td

np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
td.set_test_data(
    data_size=10000, 
    start_time=datetime.datetime(2023, 3, 21, 12, 24).timestamp(), 
    moving_av=True)

# def piecewise(self, x, params):
#     self.params = params
#     xk, yk = _Piecewise._approx(x, *params)

#     funclist = np.array([], dtype='void')
#     condlist = []
    
#     def l(x):
#         for k in range(len(xk)): 
#             if (min(x) <= xk[k]) & (xk[k] <= max(x)):
#                 break
#         if(k == len(xk) - 1):
#             k -= 1
#         # print(f'k: {k}; x: {x}')
#         # import pdb; pdb.set_trace()
#         return (yk[k] + ((yk[k+1] - yk[k]) / (xk[k+1] - xk[k])) * (x - xk[k]))

#     for k in range(self.n + 1):
#         func = lambda x: l(x)
#         funclist = np.append(funclist, func)
#         cond = (xk[k] <= x) & (x <= xk[k+1])
#         condlist.append(cond)
    
#     condlist = np.array(condlist)
    
#     return np.piecewise(x, condlist, funclist)


class Splines:
    def __init__(self, x, y, scale_x=1e-5, number_pieces=10, k=1):
        self.x = x if isinstance(x, np.ndarray) else np.array(x, dtype='float64')
        self.scale_x = scale_x
        self.x = self.x * self.scale_x
        self.y = y if isinstance(y, np.ndarray) else np.array(y, dtype='float64')
        self.y = y if isinstance(y, np.ndarray) else np.array(y, dtype='float64')
        self.n = number_pieces
        self.k = k
        self.params = None
        
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
    
    def _knots(self, params):
        xk = np.array(params[:self.n])
        xk = np.insert(xk, 0, self.x[0])
        xk = np.append(xk, self.x[-1])
        yk = np.array(params[self.n:]) 
        xy = [list(x) for x in zip(*sorted(zip(xk, yk), key=itemgetter(0)))]
        return np.array(xy[0]), np.array(xy[1])
    
    def knots(self, params=None):
        if params is None:
            params = self.params
        xk, yk = self._knots(params)
        return xk / self.scale_x, yk

    def _approx(self, x, params):
        if params is None:
            params = self.params
        xk, yk = self._knots(params)
        bspl = splrep(xk, yk, k=self.k, s=0)       
        spl = BSpline(*bspl)
        return spl(x)
    
    def approx(self, x, params=None): 
        if params is None:
            params = self.params
        xk, yk = self._knots(params)
        bspl = list(splrep(xk, yk, k=self.k, s=0))
        bspl[0] = bspl[0] / self.scale_x
        spl = BSpline(*bspl)
        return spl(x)

    def param_hedge(self, params):
        if params is None:
            params = self.params
        # import pdb; pdb.set_trace()
        xk = self._knots(params)[0][1:-1]
        deltax = (self.x[-1] - self.x[0]) / len(self.x) * 3
        excess = np.array([max(0, self.x[0] - min(xk) + deltax), max(0, max(xk) - self.x[-1] - deltax)])
        scale = sum(excess) / deltax
        hedge = excess * len(self.x) * scale ** 4
        # if sum(np.fabs(hedge)) > 0:
        #     import pdb; pdb.set_trace()
        return hedge

    def func(self, params=None):
        if params is not None:
            self.params = params
        err = (self.y - self._approx(self.x, params))
        err_norm = err / len(self.y) ** 0.5
        
        return np.append(err_norm, self.param_hedge(params))
    
    def accuracy(self, params=None):
        if params is None:
            params = self.params
        y = self._approx(self.x, params)
        err = (sum((self.y - y) ** 2) / len(self.x)) ** 0.5
        # import pdb; pdb.set_trace()
        mean = sum(np.fabs(y)) / len(self.x)
        return err / mean


class LeastSq:
    def __init__(self, func_class):
        self.func_class = func_class
    
    def run(self):
        p, e = spo.leastsq(self.func_class.func, self.func_class.param_0())
        self.func_class.params = p
        return p, e


def main():
    LIMITS = (0, 1250)
    SHIFT = 0
    DATA = td.DATA[LIMITS[0] + SHIFT: LIMITS[1] + SHIFT]
    FILTER = co.Savgol_filter(window=10, order=2)

    CNDL_COUNT = np.array([i + SHIFT for i in range(len(DATA) + SHIFT)], dtype='float64')
    VALUE = np.array([values[1][0] for values in DATA])
    filtered = FILTER.filter(VALUE)

    optimizer = LeastSq(Splines(CNDL_COUNT, VALUE, scale_x=1e-5, number_pieces=17))
    p, e = optimizer.run()
    # print(p)
    # print(optimizer.func_class.param_0())
    
    clazz = optimizer.func_class
    print(f'accuracy: {clazz.accuracy() * 100}%')
    plt.plot(CNDL_COUNT, VALUE, color='green', label='data')
    plt.plot(CNDL_COUNT, clazz.approx(CNDL_COUNT, p), color='blue', label='approx')
    plt.scatter(*clazz.knots(clazz.param_0()), color='black', label='param start')
    plt.scatter(*clazz.knots(p), color='orange', label='param final')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
