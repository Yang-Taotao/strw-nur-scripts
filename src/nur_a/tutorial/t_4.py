"""
Scripts for tutorial session 4
"""

# %%
import numpy as np

# %%
# question 1
# q1 const

bins = 10000
min_x2, max_x2 = 1, 5
min_sinx, max_sinx = 0, np.pi

# q1 temp func
x2 = lambda x: x**2
sinx = lambda x: np.sin(x)
complicated = lambda x: 3 * np.exp(-2 * x) + (x / 10) ** 4

# q1 func name
fn_name_x2 = "x2"
fn_name_sinx = "sin(x)"
fn_name_complicated = "3e^(-2x)+(x/10)^4"


# q1 func
def int_trap(min_val, max_val, bins, fn) -> np.float64:
    """trapezoid rule integral calculator"""
    # slides nur4 p6, trapezoid rule and extension
    x = np.linspace(start=min_val, stop=max_val, num=bins)
    delta_x = (max_val - min_val) / (bins - 1)

    result = delta_x * 0.5 * (fn(x[0]) + fn(x[-1]) + 2 * sum(fn(x[1:-1])))

    return result


def int_simp(min_val, max_val, bins, fn) -> np.float64:
    """simpson rule integral calculator"""
    # slides nur4 p8-11, extended simpson's rule, open formula, romberg at N=2N
    s0 = int_trap(min_val, max_val, bins, fn)
    s1 = int_trap(min_val, max_val, 2 * bins, fn)

    result = (4 / 3) * s1 - (1 / 3) * s0

    return result


# %%
# q1 func call
result_trap_x2 = int_trap(min_x2, max_x2, bins, x2)
result_trap_sinx = int_trap(min_sinx, max_sinx, bins, sinx)
result_simp_x2 = int_simp(min_x2, max_x2, bins, x2)
result_simp_sinx = int_simp(min_sinx, max_sinx, bins, sinx)
# q1 result print
print(
    f"Trapezoid integral complete for \int_{min_x2:.2f}^{max_x2:.2f} {fn_name_x2} dx with bin = {bins}. Value = {result_trap_x2:.6f}."
)
print(
    f"Trapezoid integral complete for \int_{min_sinx:.2f}^{max_sinx:.2f} {fn_name_sinx} dx with bin = {bins}. Value = {result_trap_sinx:.6f}."
)
print(
    f"Simpson integral complete for \int_{min_x2:.2f}^{max_x2:.2f} {fn_name_x2} dx with bin = {bins}. Value = {result_trap_x2:.6f}."
)
print(
    f"Simpson integral complete for \int_{min_sinx:.2f}^{max_sinx:.2f} {fn_name_sinx} dx with bin = {bins}. Value = {result_trap_sinx:.6f}."
)

# %%
