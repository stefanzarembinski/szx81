import datetime
import sys
import math
import numpy as np
# import warnings
# warnings.filterwarnings("error")
import matplotlib.pyplot as plt
from operator import itemgetter
import scipy.optimize as spo
from scipy.interpolate import splrep, BSpline

import core.core as co
import core.hist_data as td
import core.leastsqbound as lb

np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})


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
    def __init__(self, x, y, y_raw, scale_x=1e-5, number_pieces=10, k=1):
        self.x = x if isinstance(x, np.ndarray) else np.array(x, dtype='float64')
        self.scale_x = scale_x
        self.x = self.x * self.scale_x
        self.y = y if isinstance(y, np.ndarray) else np.array(y, dtype='float64')
        self.y_raw = y_raw if isinstance(y_raw, np.ndarray) else np.array(y_raw, dtype='float64')
        self.n = int(number_pieces)
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
            raise Exception(f'Parameter 0 failure! Too many pieces ({self.n})?, sep: {sep}')
        
        ik.append(len(self.x) - 1)
        xk.append(x_1)
        param = xk[1:-1]

        yk = []
        for i in ik:
            yk.append(self.y[i])
            param.append(self.y[i])
        return np.array(param)
    
    def _knots(self, params):
        params = params if type(params) is np.ndarray else params.x
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
    
    def temperature(self, params=None):
        if params is None:
            params = self.params
        noise = self.func(params, y_row=self.y_raw)
        xk, _ = self._knots(params)
        indx = [(np.abs(self.x - xk[i])).argmin() for i in range(len(xk))]
        tk = []
        for i in range(1, len(xk)):
            dist = math.fabs(indx[i-1] - indx[i])
            if dist == 0:
                tk.append(.0)
                continue
            tk.append((sum(noise[indx[i-1]: indx[i]] ** 2) / dist)** .5)
        # import pdb; pdb.set_trace()
        return np.array(tk)

    def _approx(self, x, params): 
        if params is None:
            params = self.params
        xk, yk = self._knots(params)
        bspl = splrep(xk, yk, k=self.k, s=0)       
        spl = BSpline(*bspl)
        self.params = params
        return spl(x)
    
    def approx(self, x, params=None): 
        if params is None:
            params = self.params
        xk, yk = self._knots(params)
        bspl = list(splrep(xk, yk, k=self.k, s=0))
        bspl[0] = bspl[0] / self.scale_x
        spl = BSpline(*bspl)
        return spl(x)
    
    def bounds(self):
        deltax = 0 # (self.x[-1] - self.x[0]) / len(self.x) * 3
        bounds = [(self.x[0] + deltax, self.x[-1] - deltax)] * self.n
        return bounds + [(None, None)] * (self.n + 2)
         
    def accuracy(self, params=None):
        if params is None:
            params = self.params
        y = self._approx(self.x, params)
        err = (sum((self.y - y) ** 2) / len(self.x)) ** 0.5
        # import pdb; pdb.set_trace()
        mean = sum(np.fabs(y)) / len(self.x)
        return err / mean

    def func(self, params, x=None, y=None, y_row=None):
        if y_row is None:
            return self.y - self._approx(self.x, params)
        return y_row - self._approx(self.x, params)
    

class Fitter:
    def __init__(self, func_class):
        self.func_class = func_class
    
    def run(self):
        try:
            p, _ = lb.leastsqbound(
                func=self.func_class.func,
                x0=self.func_class.param_0(),
                args=(self.func_class.x, self.func_class.y),
                bounds=self.func_class.bounds()
            )
        except RuntimeWarning as ex:
            # p, _ = lb.leastsqbound(
            #     func=self.func_class.func,
            #     x0=self.func_class.params,
            #     args=(self.func_class.x, self.func_class.y)
            # )
            import pdb; pdb.set_trace()
            p = self.func_class.params



        # def func(params):
        #     return sum(self.func_class.func(params) ** 2)
        # p = spo.minimize(
        
        #     fun=func, 
        #     x0=self.func_class.param_0(),
        #     bounds=self.func_class.bounds())
        # p = p if type(p) == np.ndarray else p.x

        self.func_class.params = p
        return p

def piecewise(
        value, filter=co.Savgol_filter(window=50, order=8),
        scale_x=1e-6, number_pieces=10, k=1):
    filtered = value
    if filter is not None:
        filtered = filter.filter(value)
    piecewise = Fitter(Splines(
        x=np.arange(-len(value) + 1, 1), 
        y=filtered, y_raw=value,
        scale_x=scale_x, number_pieces=number_pieces, k=k) )
    piecewise.run()
    return piecewise.func_class

def main():
    td.set_hist_data(
        data_count=10000, 
        start_time=datetime.datetime(2023, 3, 21, 12, 24).timestamp(), 
        moving_av=True)
    
    value = co.Savgol_filter(window=15, order=8).filter([_[0] for _ in td.VALUE][0: 120])
    clazz = piecewise(value, filter=None, number_pieces=10, k=1)
    
    cndl_count = np.arange(-len(value) + 1, 1)
    plt.plot(cndl_count, value, color='green', label='data', linewidth='0.5')
    plt.plot(cndl_count, clazz.approx(cndl_count), color='blue', label='approx')
    plt.scatter(*clazz.knots(), color='orange', label='param final')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
