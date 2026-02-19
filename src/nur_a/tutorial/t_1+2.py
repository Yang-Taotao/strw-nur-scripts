"""
Scripts for tutorial session 1+2
"""

# %%
# imports
import numpy as np
import math
import timeit
from matplotlib.image import imread


# %%
# question 1
# q1 - func
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


# %%
# q1 - func call
q1_result = compare_errors()
print(q1_result)

# %%
# question 2
# q2 - const assignment
G = 6.67 * 10 ** (-11)
C = 3 * 10 ** (8)
C_INV = 1 / C
C_INV2 = C_INV**2


# %%
# q2 - func
def bh_gen(mean: int = 1000000, stdev: int = 100000, size: int = 10000) -> np.ndarray:
    """create `size` number of bh from gaussian distribution"""
    return np.random.normal(loc=mean, scale=stdev, size=size)


def rs_div(mass: np.ndarray):
    """directly calculate Schwarzschild radius rs with dividing c**2"""
    return 2 * G * mass / C**2


def rs_mult(mass: np.ndarray):
    """directly calculate Schwarzschild radius rs with multiplying (1/c)**2"""
    return 2 * G * mass * C_INV2


def rs_compare(mean: int = 1000000, stdev: int = 100000, size: int = 10000):
    """compare rs comp time with timeit"""
    masses = bh_gen()
    t_div = timeit.timeit(lambda: rs_div(masses), number=size)
    t_mult = timeit.timeit(lambda: rs_mult(masses), number=size)

    print(f"rs_div at {t_div}")
    print(f"rs_mult at {t_mult}")

    return None


# %%
# q2 - func call
rs_compare()

# %%
# question 3
# q3 - func
# def img_interpolation():

# def linear_interpolator(x1, x2):
#     l_lim = 0
#     r_lim = len(x2) - 1
#     while r_lim - l_lim > 1:
#         mid = (l_lim+r_lim) // 2
#         if x2[mid] < x1:
#             l_lim = mid
#         else:
#             r_lim = mid
#     return l_lim, r_lim

# def polynomial_interpolator():
