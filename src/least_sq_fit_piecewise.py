import numpy as np

from operator import itemgetter
import scipy.optimize as spo
from scipy.interpolate import splrep, BSpline

from core import *
import test_data as td

np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
td.set_test_data(
    data_size=10000, 
    start_time=datetime.datetime(2023, 3, 21, 12, 24).timestamp(), 
    moving_av=True)

LIMITS = (0, 250)
SHIFT = 0
DATA = td.DATA[LIMITS[0] + SHIFT: LIMITS[1] + SHIFT]
FILTER = Savgol_filter(window=10, order=2)

CNDL_COUNT = np.array([i + SHIFT for i in range(len(DATA) + SHIFT)], dtype='float64')
VALUE = np.array([values[1][0] for values in DATA])
filtered = FILTER.filter(VALUE)


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
    def __init__(self, x, y, scale_x=1e-5, number_pieces=10):
        self.x = x if isinstance(x, np.ndarray) else np.array(x, dtype='float64')
        self.scale_x = scale_x
        self.x = self.x * self.scale_x
        self.y = y if isinstance(y, np.ndarray) else np.array(y, dtype='float64')
        self.y = y if isinstance(y, np.ndarray) else np.array(y, dtype='float64')
        self.n = number_pieces
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
    
    def _knots(self):
        xk = np.array(self.params[:self.n])
        xk = np.insert(xk, 0, self.x[0])
        xk = np.append(xk, self.x[-1])
        yk = np.array(self.params[self.n:]) 
        xy = [list(x) for x in zip(*sorted(zip(xk, yk), key=itemgetter(0)))]
        return xy[0], xy[1]
    
    def knots(self, params):
        self.params = params
        xk, yk = self._knots()
        return [x / self.scale_x for x in xk], yk

    def _approx(self, x):
        xk, yk = self._knots()
        bspl = splrep(xk, yk, k=1, s=0)       
        spl = BSpline(*bspl)
        return spl(x)
    
    def approx(self, x, params):
        self.params = params  
        xk, yk = self._knots()
        bspl = list(splrep(xk, yk, k=1, s=0))
        bspl[0] = bspl[0] / self.scale_x
        spl = BSpline(*bspl)
        return spl(x)

    def param_hedge(self):
        # import pdb; pdb.set_trace()
        xk = self._knots()[0][1:-1]
        # return np.array([self.x[0] - min(xk), max(xk) - self.x[-1]])

    def func(self, params):
            self.params = params
            self.param_hedge()
            return self.y - self._approx(self.x)

class LeastSq:
    def __init__(self, func_class):
        self.func_class = func_class
    
    def run(self):
        return spo.leastsq(self.func_class.func, self.func_class.param_0())
    
optimizer = LeastSq(Splines(CNDL_COUNT, VALUE, scale_x=1e-5, number_pieces=17))
p, e = optimizer.run()
print(p)
print(optimizer.func_class.param_0())

clazz = optimizer.func_class
plt.plot(CNDL_COUNT, VALUE, color='green', label='data')
plt.plot(CNDL_COUNT, clazz.approx(CNDL_COUNT, p), color='blue', label='approx')
plt.scatter(*clazz.knots(clazz.param_0()), color='black', label='param start')
plt.scatter(*clazz.knots(p), color='orange', label='param final')

plt.legend()
plt.show()