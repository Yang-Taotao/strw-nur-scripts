# %%
import numpy as np
import math


# %% question 1
def sinc_x_lib(x):
    """np implementation of sinc"""
    return np.sin(x) / x


def sinc_x_power(x, n):
    """ps implementation of sinc"""
    n_ary = np.arange(0, n, 1)
    top = np.sum((-1) ** n * x ** (2 * n))
    bot = math.factorial(2 * n + 1)
    return top / bot


def numerical_errors(x, n):
    """compare np sinc and ps sinc"""
    return sinc_x_power(x, n) - sinc_x_lib(x)


def compare_errors(x: int = 7, n_range: int = 10):
    """do comparison with multiple choices of n"""
    result = [numerical_errors(x, n_arg) for n_arg in range(1, n_range + 1)]
    return result


q1_result = compare_errors()
print(q1_result)

# %% question 2
G = 6.67 * 10 ** (-11)
C = 3 * 10 ** (8)


def bh_gen(mean: int = 1000000, stdev: int = 100000, size: int = 10000):
    """create `size` number of bh from gaussian distribution"""
    return np.random.normal(loc=mean, scale=stdev, size=size)
