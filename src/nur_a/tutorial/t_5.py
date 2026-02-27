"""
Scripts for tutorial session 4
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# question 1


def fxd0(x):
    """original fx"""
    return x**2 * np.sin(x)


def fxd1(x):
    """detivative of x**2 * np.sin(x)"""
    return 2 * x * np.sin(x) + x**2 * np.cos(x)


def analytic_diff(min_val: float = 0.0, max_val: float = 2 * np.pi, steps: int = 200):
    """get analytic solution val"""
    x = np.linspace(min_val, max_val, steps, dtype="float64")
    y = fxd1(x)
    return y


def central_diff(
    min_val: float = 0.0, max_val: float = 2 * np.pi, steps: int = 200, h: float = 1.0
):
    """central diff method for differentiation"""
    x = np.linspace(min_val, max_val, steps, dtype="float64")
    # y = [(fxd0(x[i] + h) - (fxd0(x[i] - h))) / 2 * h for i in range(steps)]
    x_plus = x + h
    x_minu = x - h
    y = (1 / (2 * h)) * (fxd0(x_plus) - fxd0(x_minu))
    return y


# def ridder(m: int = 5, h: float = 0.1, d: float = 2.0):
#     """ridder method"""
#     for i in range(m):
#         for j in range(m):


def q1_plot(
    y0,
    y1,
    y2,
    y3,
    y4,
    min_val: float = 0.0,
    max_val: float = 2 * np.pi,
    steps: int = 200,
):
    """hard coded plotter for simplicity"""
    x = np.linspace(min_val, max_val, steps, dtype="float64")
    fig, ax = plt.subplots()
    ax.plot(x, y0, label="analytical")
    ax.plot(x, y1, label="h=1.0")
    ax.plot(x, y2, label="h=0.1")
    ax.plot(x, y3, label="h=0.01")
    ax.plot(x, y4, label="h=0.001")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()
    plt.close()


# %%
# q1 call

# fxd1_plot()
y0 = analytic_diff()
y1 = central_diff(h=1.0)
y2 = central_diff(h=0.1)
y3 = central_diff(h=0.01)
y4 = central_diff(h=0.001)
q1_plot(y0, y1, y2, y3, y4)

# %%
# question 2

fn_theta0 = lambda x: np.pi * x
fn_phi0 = lambda x: 2 * np.pi * x

fn_theta1 = lambda x: np.arccos(1 - 2 * x)
fn_phi1 = lambda x: 2 * np.pi * x


def gen_3d_rand(fn_theta, fn_phi, min_val: int = 0, max_val: int = 1, npts: int = 5000):
    """gen uniform random num on 3d space for some fn"""
    r = np.ones(npts)
    theta = fn_theta(np.random.uniform(min_val, max_val, npts))
    phi = fn_phi(np.random.uniform(min_val, max_val, npts))

    return r, theta, phi


def rand_3d_plot():
    r0, theta0, phi0 = gen_3d_rand(fn_theta0, fn_phi0)
    r1, theta1, phi1 = gen_3d_rand(fn_theta1, fn_phi1)

    x0, y0, z0 = (
        r0 * np.sin(phi0) * np.cos(theta0),
        r0 * np.sin(phi0) * np.sin(theta0),
        r0 * np.cos(phi0),
    )

    x1, y1, z1 = (
        r1 * np.sin(phi1) * np.cos(theta1),
        r1 * np.sin(phi1) * np.sin(theta1),
        r1 * np.cos(phi1),
    )

    fig1 = plt.figure()
    ax1 = plt.axes(projection="3d")
    ax1.scatter(x0, y0, z0, c="r")
    plt.show()
    plt.close()

    fig2 = plt.figure()
    ax2 = plt.axes(projection="3d")
    ax2.scatter(x1, y1, z1, c="g")
    plt.show()
    plt.close()


rand_3d_plot()
