"""
Scripts for assignment 1 question 2
"""

import os
import sys
import timeit

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.size"] = 20
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20


def load_data():
    """
    Function to load the data from Vandermonde.txt.

    Returns
    ------------
    x (np.ndarray): Array of x data points.

    y (np.ndarray): Array of y data points.
    """
    data = np.genfromtxt(
        "./data/vandermonde.txt",
        comments="#",
        dtype=np.float64,
    )
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def construct_vandermonde_matrix(x: np.ndarray) -> np.ndarray:
    """
    Construct the Vandermonde matrix V with V[i,j] = x[i]^j.

    Parameters
    ----------
    x : np.ndarray, x-values.

    Returns
    -------
    V : np.ndarray, Vandermonde matrix.
    """
    # init vandermonde mat at shape (len(x), len(x))
    # use consistent float64 dtype as load_data()
    n = len(x)
    v_mat = np.zeros((n, n), dtype=np.float64)

    # assign val at elements with V(i,j) as x_i^j, first col == 1 by def
    # 1st col -> 0 idx == [1, 1, ...] Transpose -> V(0,0)=1.0
    # 2nd col -> 1 idx == [x_0, x_1, ...] Transpose -> V(1,0)=1.0*x_0
    # 3rd col -> 2 idx == [x_0**2, x_1**2, ...] Transpose -> V(2,0)=(1.0*x_0)*x_0
    # efficient through col assignment by doing col_(j) = x**(j-1)
    # alternative -> col_(j) = x * col_(j-1)
    # with j=0 col assignment -> reducing n(operations) on each row with
    v_mat[:, 0] = np.float64(1.0)
    # now loop over the rest of the col
    for j in range(1, n):
        v_mat[:, j] = x * v_mat[:, j - 1]

    return v_mat


def LU_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Perform LU decomposition.

    The lower-triangular matrix (L) is stored in the lower part of A (the diagonal elements are assumed =1),
    while the upper-triangular matrix (U) is stored on and above the diagonal of A.

    Parameters
    ----------
    A : np.ndarray
        Matrix to decompose.

    Returns
    -------
    A : np.ndarray
        Decomposed array.
    """
    n_row, n_col = A.shape
    # error if not square
    if n_row != n_col:
        raise ValueError(
            f"Abort with non-square matrix. Current shape ({A.shape[0]}, {A.shape[1]})."
        )
    # error if singular
    if n_row == n_col == 1:
        raise ValueError(
            f"Abort with a singular matrix. Current shape ({A.shape[0]}, {A.shape[1]})."
        )

    # do gaussian elimination with mat element A(i,j)
    # 1st row, 1st col item is A(1, 1) in mat, code at 0 idx equivalent to A[0, 0]
    # L -> terms with i > j
    # U -> terms with i <= j
    # LU -> combined L and U into one mat
    # LU through crout's -> pivot -> loop over col k -> then row i
    # last diag term does not need pivot
    for k in range(n_col - 1):  # k in [0, n_col-1)
        pivot = A[k, k]
        # break if zero pivot
        if pivot == np.float64(0.0):
            raise ValueError(f"Found zero pivot, pivot({k}, {k}) = {pivot}.")
        # do operation on whole row
        for i in range(k + 1, n_col):  # i in [1, n_col)
            # L(i,k) = A(i,k)/A(k,k), i>k
            A[i, k] /= pivot
            # update row -< reduce
            A[i, k + 1 :] -= A[i, k] * A[k, k + 1 :]

    return A

    # general form of A for LU follows
    # A = mat(A_ij)
    # if A x = b, with A = LU
    # LU x = b
    # we can sub with y = U x -> L y = b
    # we now have 2 sets of eqs 1) y = U x, 2) L y = b
    # we can now solve for y and x


def forward_substitution_unit_lower(LU: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve L*y = b using forward substitution,
    where L is the lower-triangular matrix.

    Parameters
    ----------
    LU : np.ndarray
        LU matrix from LU_decomposition.
    b : np.ndarray
        Right-hand side.

    Returns
    -------
    y : np.ndarray
        Solution vector.
    """
    # forward sub
    # lecture 3 p11
    # L * y = b
    # y has shape agreement with b
    y = np.zeros(len(b), dtype=np.float64)

    # y_i = (1 / L_ii) * (b_i - sum_{j=0}^{i-1}(y_j * L_ij))
    # because of LU mat instead of L and U mat, we can safely set L_ii = 1
    #
    # y[0] = b[0]
    # y[1] = b[1] - (LU[1,0] * y[0])
    # y[2] = b[2] - (LU[2,0] * y[0] + LU[2,1] * y[1])
    for i in range(len(y)):
        the_sum = np.float64(0.0)
        for j in range(i):
            the_sum += LU[i, j] * y[i]
        y[i] = b[i] - the_sum

    return y


def backward_substitution_upper(LU: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve U*c = y using backward substitution,
    where U is the upper-triangular matrix.

    Parameters
    ----------
    LU : np.ndarray
        LU matrix from LU_decomposition.
    y : np.ndarray
        Right-hand side.

    Returns
    -------
    c : np.ndarray
        Solution vector.
    """
    # backward sub
    # lecture 3 p11 -> U*x = y -> use c notations here
    # make results ary like in forward sub
    c = np.zeros(len(y), dtype=np.float64)

    # c_i = (1/U_ii)*(y_i-sum_{j=i+1}^{n-1}(U_ij*c_j))
    # for j in [i+1, n-1) -> n terms -> idx n-1 = idx -1 -> idx j as [i+1, -1, -1]
    #
    # just like in forward sub, we use a LU mat instead of a separate U
    # U_ij = LU_ij since j >= i
    #
    # set n=3
    # c[0] = (1/LU[0,0])*(y[0]-(LU[0,2]*c[2] + LU[0,1]*c[1]))
    # c[1] = (1/LU[1,1])*(y[1]-(LU[1,2]*c[2]))
    # c[2] = (1/LU[2,2])*(y[2])
    #
    # loop from last i, i = n-1 = 2 -> loop 2, 1, 0 -> range(n-1, -1, -1)
    # for j val
    # when at i = 2 -> empty j loop
    # when at i = 1 -> loop j at 2
    # when at i = 0 -> loop j at 1, 2
    for i in range(len(c) - 1, -1, -1):
        the_sum = np.float64(0.0)
        for j in range(i + 1, len(c)):
            the_sum += LU[i, j] * c[j]
        c[i] = (1 / LU[i, i]) * (y[i] - the_sum)
    return c


def vandermonde_solve_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve for polynomial coefficients c from data (x,y) using the Vandermonde matrix.

    Parameters
    ----------
    x : np.ndarray
        x-values.
    y : np.ndarray
        y-values.

    Returns
    -------
    c : np.ndarray
        Polynomial coefficients.
    """
    # TODO:
    # solve V*c = y using your LU decomposition and forward/backward substitution
    return np.zeros_like(x)  # Replace with your solution


def evaluate_polynomial(c: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Evaluate y(x) = sum_j c[j] * x^j.

    Parameters
    ----------
    c : np.ndarray
        Polynomial coefficients.
    x_eval : np.ndarray
        Evaluation points.

    Returns
    -------
    y_eval : np.ndarray
        Polynomial values.
    """
    # TODO:
    # evaluate the polynomial at x_eval using the coefficients c from vandermonde_solve_coefficients
    return np.zeros_like(x_eval)  # Replace with your results


def neville(x_data: np.ndarray, y_data: np.ndarray, x_interp: float) -> float:
    """
    Function that applies Nevilles algorithm to calculate the function value at x_interp.

    Parameters
    ------------
    x_data (np.ndarray): Array of x data points.
    y_data (np.ndarray): Array of y data points.
    x_interp (float): The x value at which to interpolate.

    Returns
    ------------
    float: The interpolated y value at x_interp.
    """
    # TODO:
    # write your Neville's algorithm
    return 0


# you can merge the function below with LU_decomposition to make it more efficient
def run_LU_iterations(
    x: np.ndarray,
    y: np.ndarray,
    iterations: int = 11,
    coeffs_output_path: str = "Coefficients_per_iteration.txt",
):
    """
    Iteratively improves computation of coefficients c.

    Parameters
    ----------
    x : np.ndarray
        x-values.
    y : np.ndarray
        y-values.
    iterations : int
        Number of iterations.
    coeffs_output_path : str
        File to write coefficient values per iteration.

    Returns
    -------
    coeffs_history :
        List of coefficient vectors.
    """
    # TODO:
    # Implement an iterative improvement for computing the coefficients c,
    # and save the coefficients at each iteration to coeffs_output_path.
    return [np.zeros_like(x) for _ in range(iterations)]  # Replace with your solution


def plot_part_a(
    x_data: np.ndarray,
    y_data: np.ndarray,
    coeffs_c: np.ndarray,
    plots_dir: str = "./plots",
) -> None:
    """
    Ploting routine for part (a) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    coeffs_c : np.ndarray
        Polynomial coefficients c.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """
    xx = np.linspace(x_data[0], x_data[-1], 1001)
    yy = evaluate_polynomial(coeffs_c, xx)
    y_at_data = evaluate_polynomial(coeffs_c, x_data)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0)
    axs[0].plot(xx, yy, linewidth=3)
    axs[0].set_xlim(
        np.floor(xx[0]) - 0.01 * (xx[-1] - xx[0]),
        np.ceil(xx[-1]) + 0.01 * (xx[-1] - xx[0]),
    )
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(["data", "Via LU decomposition"], frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")
    axs[1].plot(x_data, np.abs(y_data - y_at_data), linewidth=3)

    plt.savefig(os.path.join(plots_dir, "a1q2_vandermonde_sol_2a.pdf"))
    plt.close()


def plot_part_b(
    x_data: np.ndarray,
    y_data: np.ndarray,
    plots_dir: str = "./plots",
) -> None:
    """
    Ploting routine for part (b) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """
    xx = np.linspace(x_data[0], x_data[-1], 1001)
    yy = np.array([neville(x_data, y_data, x) for x in xx], dtype=np.float64)
    y_at_data = np.array([neville(x_data, y_data, x) for x in x_data], dtype=np.float64)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0)
    axs[0].plot(xx, yy, linestyle="dashed", linewidth=3)
    axs[0].set_xlim(
        np.floor(xx[0]) - 0.01 * (xx[-1] - xx[0]),
        np.ceil(xx[-1]) + 0.01 * (xx[-1] - xx[0]),
    )
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(["data", "Via Neville's algorithm"], frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")
    axs[1].plot(x_data, np.abs(y_data - y_at_data), linestyle="dashed", linewidth=3)

    plt.savefig(os.path.join(plots_dir, "a1q2_vandermonde_sol_2b.pdf"))
    plt.close()


def plot_part_c(
    x_data: np.ndarray,
    y_data: np.ndarray,
    coeffs_history: list[np.ndarray],
    iterations_num: list[int] = [0, 1, 10],
    plots_dir: str = "./plots",
) -> None:
    """
    Ploting routine for part (c) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    coeffs_history : list[np.ndarray]
        Coefficients per iteration.
    iterations_num : list[int]
        Iteration numbers to plot.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """

    linstyl = ["solid", "dashed", "dotted"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    xx = np.linspace(x_data[0], x_data[-1], 1001)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0, color="black", label="data")

    for i, k in enumerate(iterations_num):
        if k >= len(coeffs_history):
            continue
        c = coeffs_history[k]
        yy = evaluate_polynomial(c, xx)
        y_at_data = evaluate_polynomial(c, x_data)
        diff = np.abs(y_at_data - y_data)

        axs[0].plot(
            xx,
            yy,
            linestyle=linstyl[i],
            color=colors[i],
            linewidth=3,
            label=f"Iteration {k}",
        )
        axs[1].plot(x_data, diff, linestyle=linstyl[i], color=colors[i], linewidth=3)

    axs[0].set_xlim(
        np.floor(xx[0]) - 0.01 * (xx[-1] - xx[0]),
        np.ceil(xx[-1]) + 0.01 * (xx[-1] - xx[0]),
    )
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")

    plt.savefig(os.path.join(plots_dir, "a1q2_vandermonde_sol_2c.pdf"))
    plt.close()


def main():
    os.makedirs("./plots", exist_ok=True)
    x_data, y_data = load_data()

    # compute times
    number = 10

    t_a = (
        timeit.timeit(
            stmt=lambda: vandermonde_solve_coefficients(x_data, y_data),
            number=number,
        )
        / number
    )

    xx = np.linspace(x_data[0], x_data[-1], 1001)
    t_b = (
        timeit.timeit(
            stmt=lambda: np.array(
                [neville(x_data, y_data, x) for x in xx], dtype=np.float64
            ),
            number=number,
        )
        / number
    )

    t_c = (
        timeit.timeit(
            stmt=lambda: run_LU_iterations(x_data, y_data, iterations=11),
            number=number,
        )
        / number
    )

    # write all timing
    with open("./output/a1q2_execution_times.txt", "w", encoding="utf-8") as f:
        f.write(f"\\item Execution time for part (a): {t_a:.5f} seconds\n")
        f.write(f"\\item Execution time for part (b): {t_b:.5f} seconds\n")
        f.write(f"\\item Execution time for part (c): {t_c:.5f} seconds\n")

    c_a = vandermonde_solve_coefficients(x_data, y_data)
    plot_part_a(x_data, y_data, c_a)

    formatted_c = [f"{coef:.3e}" for coef in c_a]
    with open("./output/a1q2_coefficients_output.txt", "w", encoding="utf-8") as f:
        for i, coef in enumerate(formatted_c):
            f.write(f"c$_{i+1}$ = {coef}, ")

    plot_part_b(x_data, y_data)

    coeffs_history = run_LU_iterations(
        x_data,
        y_data,
        iterations=11,
        coeffs_output_path="./output/a1q2_coefficients_per_iteration.txt",
    )
    plot_part_c(x_data, y_data, coeffs_history, iterations_num=[0, 1, 10])


if __name__ == "__main__":
    main()
