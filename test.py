import math
import random as rand
import numpy as np
import scipy
import scipy.optimize

rng = np.random.default_rng()
x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)

xlimits = 0, 3
xcount = 50
x = [xlimits[0] + i * (xlimits[1] - xlimits[0]) / xcount for i in range(xcount)]
y = [math.exp(-xx**2) + 0.05 * rand.gauss(0, 1) for xx in x]
# y = [xx**4 + 0.1 * rand.gauss(0, 1) for xx in x]


# Fit a polynomial of degree 2 (quadratic)
FROM = 15
TILL = 50

x_sub = x[FROM:TILL]
y_sub = y[FROM:TILL]
coefficients, residuals, rank, singular_values, rcond = np.polyfit(x_sub, y_sub, 2, full=True)
coefficients3, residuals3, rank3, singular_values3, rcond3 = np.polyfit(x_sub, y_sub, 3, full=True)
# print(coefficients)
# print(coefficients3)

def f(u):
    # global FROM
    u = int(u)
    x = x_sub[0:u]
    y = y_sub[0:u]
    # import pdb; pdb.set_trace()
    coefficients, residuals, rank, singular_values, rcond = np.polyfit(x, y, 2, full=True)
    return residuals[0] / u

res = scipy.optimize.minimize_scalar(f, bounds=(5, len(x_sub)), method='bounded')
print("optimal residuum:", res.fun)
print("optimal length: ", round(res.x))

class Data:
    def __init__(self):
        rng = np.random.default_rng()
        self.len = 100
        x = np.linspace(-3, 3, self.len)
        y = np.exp(-x**2) + 0.1 * rng.standard_normal(self.len)
        self.xy = [[x[i], y[i]] for i in range(self.len)]

    def data(self):
        i = 0
        while i < self.len:
            yield self.xy[i]
            i += 1

data = Data()

def nextToken():
    x = []
    y = []
    for i in range(4):
        xy = next(data)
        x.append(xy[0])
        y.append(xy[1])
        oefficients, residuals, rank, singular_values, rcond = np.polyfit(x, y, deg=3, full=True) 



# Fit a polynomial of degree 2 (quadratic)
coefficients, residuals, rank, singular_values, rcond = np.polyfit(x, y, deg=10, full=True)
print(coefficients)
print(residuals)
print(rank)


# Generate polynomial function
polynomial = np.poly1d(coefficients)
print(polynomial)
