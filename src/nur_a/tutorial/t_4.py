"""
Scripts for tutorial session 4
"""

# %%
import numpy as np

# %%
# question 1
# q1 func


def x2(x):
    return x**2


def sinx(x):
    return np.sin(x)


def int_trap(min, max, bin, fn: str) -> np.float64:
    """trapezoid rule integral calculator"""
    # slides nur4 p6, trapezoid rule and extension
    x = np.linspace(start=min, stop=max, num=bin)
    delta_x = (max - min) / (bin - 1)

    if fn == "x2":
        y0, y1 = x2(x[0]), x2(x[-1])
        sum_mid = sum(x2(x[1:-1]))

    elif fn == "sinx":
        y0, y1 = sinx(x[0]), sinx(x[-1])
        sum_mid = sum(sinx(x[1:-1]))

    result = delta_x * 0.5 * (y0 + y1 + 2 * sum_mid)

    print(
        f"Trapezoid integral complete for \int_{min:.2f}^{max:.2f} {fn} dx with bin = {bin}. Value = {result:.6f}."
    )
    return result


def int_simp(min, max, bin, fn: str) -> np.float64:
    """simpson rule integral calculator"""
    # slides nur4 p8, extended simpson's rule
    x = np.linspace(start=min, stop=max, num=bin)
    delta_x = (max - min) / (bin - 1)

    if fn == "x2":
        y0, y1 = x2(x[0]), x2(x[-1])
        sum_mid = sum(4 * x2(x[1:-1:2])) + sum(2 * x2(x[2:-2:2]))

    elif fn == "sinx":
        y0, y1 = sinx(x[0]), sinx(x[-1])
        sum_mid = sum(4 * sinx(x[1:-1:2])) + sum(2 * sinx(x[2:-2:2]))

    result = (delta_x / 3) * (y0 + y1 + sum_mid)

    print(
        f"Simpson integral complete for \int_{min:.2f}^{max:.2f} {fn} dx with bin = {bin}. Value = {result:.6f}."
    )

    return result


# %%
# q1 func call
int_trap(1, 5, 10000, "x2")
int_trap(0, np.pi, 10000, "sinx")

int_simp(1, 5, 10000, "x2")
int_simp(0, np.pi, 10000, "sinx")
