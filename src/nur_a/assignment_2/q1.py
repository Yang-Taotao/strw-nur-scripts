"""
Scripts for assignment 2 question 1
"""

# imports
import numpy as np
import matplotlib.pyplot as plt

# ==================================================================================== #
# RNG - START

# local const
MASK_64 = (1 << 64) - 1
MASK_32 = (1 << 32) - 1
XOR_A = 21
XOR_B = 35
XOR_C = 4
MWC_A = 4294957665
NUM_2_64 = 2**64


def xor_ops(state: int) -> int:
    """Do 64-bit XOR shift for rng gen"""
    # do bit-wise ops through XOR-shift
    # 1) x = x^(x>>a)
    # 2) x = x^(x<<b)
    # 3) x = x^(x>>c)
    # we reapply 64 bit mask at every op to ensure consistency
    state ^= state >> XOR_A
    state &= MASK_64
    state ^= state << XOR_B
    state &= MASK_64
    state ^= state >> XOR_C
    state &= MASK_64

    return state


def mwc_ops(state: int) -> int:
    """Do multiply with carry with base 2**32"""
    # find lowest and highest bits in this state
    lowest = state & MASK_32
    highest = state >> 32
    # combine into new state
    # apply 32-bit mask for consistency
    state = MWC_A * lowest + highest
    state &= MASK_64

    return state


def rng_gen(state_xor: int, state_mwc: int) -> tuple[int, int, int]:
    """Evolve rng with xor and mwc methods"""
    # get next state from current state
    nxt_state_xor = xor_ops(state_xor)
    nxt_state_mwc = mwc_ops(state_mwc)

    # use lower 32 bits for mwc
    nxt_state_mwc_32 = nxt_state_mwc & MASK_32

    # merge results
    rng_val = nxt_state_xor ^ nxt_state_mwc_32

    return nxt_state_xor, nxt_state_mwc, rng_val


class NURARNG:
    """Custom RNG for NURA assignment"""

    # seed vs state
    # seed -> something you start with, it doesn't change
    # state -> computed based off of seed, it changes per niter

    def __init__(self, seed: int = 42) -> None:
        """Seed assignment init"""
        # avoid zero seed for xor
        if seed == 0:
            seed = 42

        seed &= MASK_64
        self.state_xor = seed
        self.state_mwc = seed

    def nxt_iter(self) -> int:
        """Get to next iter, given current iter results, return rng_val"""
        nxt_state_xor, nxt_state_mwc, rng_val = rng_gen(self.state_xor, self.state_mwc)
        self.state_xor = nxt_state_xor
        self.state_mwc = nxt_state_mwc
        return rng_val

    def rand(self) -> float:
        """Return normalized rng_val similar to np.random.rand()"""
        return self.nxt_iter() / NUM_2_64


rng = NURARNG()


def plot_rng_uniformity(n_bins: int = 100, n_samples: int = 10000) -> None:
    """Plot rng results"""
    # sample gen
    samples = np.array([rng.rand() for _ in range(n_samples)])

    fig, ax = plt.subplots()
    ax.hist(samples, bins=n_bins, density=True, alpha=0.6, label="Samples")
    ax.axhline(1.0, color="r", linestyle="--", label="Uniform PDF")

    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.legend()

    fig.savefig("./plots/a2q1_rng_uniformity_test.png", dpi=600)
    plt.close(fig)


# RNG - END
# ==================================================================================== #
# NDEN - START


def n(x: float, A: float, Nsat: float, a: float, b: float, c: float) -> float:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : float
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    return A * Nsat * (x / b) ** (a - 3) * np.exp(-((x / b) ** c))


# NDEN - END
# ==================================================================================== #
# INTG - START


# Below we provide a template for romberg integration
# You can implement this
# or use another integration method based on some form of Richardson extrapolation
def romberg_integrator(
    func: callable, bounds: tuple, order: int = 5, err: bool = False, args: tuple = ()
) -> float:
    """
    Romberg integration method

    Parameters
    ----------
    func : callable
        Function to integrate.
    bounds : tuple
        Lower- and upper bound for integration.
    order : int, optional
        Order of the integration.
        The default is 5.
    err : bool, optional
        Whether to retun first error estimate.
        The default is False.
    args : tuple, optional
        Arguments to be passed to func.
        The default is ().

    Returns
    -------
    float
        Value of the integral. If err=True, returns the tuple
        (value, err), with err a first estimate of the (relative)
        error.
    """
    # local assignment
    a, b = bounds
    h = b - a
    fa = func(a, *args)
    fb = func(b, *args)

    # romberg - init
    r = np.zeros(order)
    # get r_0
    r[0] = (0.5 * h) * (fa + fb)

    # romberg - fill 1st col
    Np = 1
    for i in range(1, order):
        # new local assignment
        the_sum = 0.0
        delta = h
        h /= 2.0
        x = a + h

        for _ in range(Np):
            the_sum += func(x, *args)
            x += delta

        r[i] = 0.5 * (r[i - 1] + delta * the_sum)
        Np *= 2

    # romberg <-> neville
    # reset assignmnet
    Np = 1
    # loop through col
    for i in range(1, order):
        Np *= 4
        # loop through row
        for j in range(0, order - i):
            r[j] = (Np * r[j + 1] - r[j]) / (Np - 1)
    # get r[0] as best estimate for integral
    result = r[0]

    # errors
    if err:
        if order > 1:
            error = np.abs(r[1] - r[0])
        else:
            error = 0.0
        return result, error

    return result


# INTG - END
# ==================================================================================== #
# SAMP - START


def sampler(
    dist: callable,
    min_val: float,
    max_val: float,
    Nsamples: int,
    args: tuple = (),
) -> np.ndarray:
    """
    Sample a distribution using sampling method of your choice - slice sampling

    Parameters
    dist : callable
        Distribution to sample
    min_val :
        Minimum value for sampling
    max_val : float
        Maximum value for sampling
    Nsamples : int
        Number of samples
    args : tuple, optional
        Arguments of the distribution to sample, passed as args to dist

    Returns
    -------
    sample: ndarray
        Values sampled from dist, shape (Nsamples,)
    """
    # choose initial x with dist(x) > 0 -> say we start with a mid point x0
    x = 0.5 * (min_val + max_val)
    # endure initial dist(x0) > 0 -> loop until it is
    while dist(x, *args) <= 0.0:
        x = min_val + (max_val - min_val) * rng.rand()

    # init samples
    samples = []
    # set step size for horizon tracing
    delta = 0.0001 * (max_val - min_val)

    # generate y ~ U(0, dist(x)) = U(0, 1)*dist(x)
    for _ in range(Nsamples):
        px = dist(x, *args)
        y = px * rng.rand()

        # get horizon bound x0, x1
        x0, x1 = x, x
        while x0 > min_val and dist(x0, *args) > y:
            x0 -= delta
        while x1 < max_val and dist(x1, *args) > y:
            x1 += delta

        # get the new sample
        while True:
            x_sample = x0 + (x1 - x0) * rng.rand()
            if dist(x_sample, *args) >= y:
                break
            # scale down to region
            if x_sample < x:
                x0 = x_sample
            else:
                x1 = x_sample

        samples.append(x_sample)

        # update x and goto next iter
        x = x_sample

    return np.array(samples)


# SAMP - END
# ==================================================================================== #
# SORT - START


def heap_ops(ary: np.ndarray, n: int, i: int) -> None:
    """Recursive heap sorting ops, build heap with size n"""
    # [2, 1, 5, 3, 7, 6, 8]
    # heap 0 1 2 3 4 5 6
    # elem 2 1 5 3 7 6 8

    # max heap at   =>  swap
    # 8                 5
    # 3   7             3   7
    # 2 1 6 5           2 1 6 /8

    # recursive
    # to max heap   =>  swap
    # 7                 5
    # 3   6             3   6
    # 2 1 5 /8          2 1 /7 /8

    # get max idx
    max_heap = i
    # get idx bounds - heap child of idx i
    l = 2 * i + 1
    r = 2 * i + 2

    # find max idx
    if n > l and ary[l] > ary[max_heap]:
        max_heap = l
    if n > r and ary[r] > ary[max_heap]:
        max_heap = r

    # swap
    if max_heap != i:
        ary[max_heap], ary[i] = ary[i], ary[max_heap]

        # recursive ops
        heap_ops(ary, n, max_heap)


def sort_array(
    arr: np.ndarray,
    inplace: bool = False,
) -> np.ndarray:
    """
    Sort a 1D array using a sorting algorithm of your choice - heap sort

    Parameters
    ----------
    arr : ndarray
        Input array to be sorted
    inplace : bool, optional
        If True, sort the array in-place
        If False, return a sorted copy

    Returns
    -------
    ary : ndarray
        Sorted array (same shape as arr)

    """
    # check inplace flag
    if inplace == False:
        ary = arr.copy()
    else:
        ary = arr

    # get array size
    n = len(ary)

    # init max heap
    for i in range((n // 2) - 1, -1, -1):
        # build heap
        heap_ops(ary, n, i)

    # get max element and swap
    for i in range(n - 1, 0, -1):
        # swap largest element with last
        ary[0], ary[i] = ary[i], ary[0]
        # build heap from current state
        heap_ops(ary, i, 0)

    return ary


def choice(arr: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Choose given number of random elements from an array, without replacement.
    Fisher-Yates shuffling

    Parameters
    ----------
    arr : ndarray
        Array to shuffle
    size : int, optional
        Number of elements to pick from array
        The default is 1

    Returns
    -------
    chosen : ndarray
        Randomly chosen elements from arr, shape (size,)
    """
    ary = arr.copy()

    if size > len(ary):
        raise ValueError(
            "Invalid slice size for choice(), slcie size must be leq than array size."
        )

    # loop over idx
    for i in range(size):
        # get a nex idx from current through rng
        mod = int((len(ary) - i) * rng.rand())
        j = i + mod
        # swap
        ary[i], ary[j] = ary[j], ary[i]

    return ary[:size]


# SORT - END
# ==================================================================================== #
# DIFF - START


def dn_dx(x: float, A: float, Nsat: float, a: float, b: float, c: float) -> float:
    """
    Analytical derivative of number density provide

    Parameters
    ----------
    x : ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float
        Same type and shape as x. Derivative of number density of
        satellite galaxies at given radius x.
    """
    # f(x) = A * Nsat * (x / b) ** (a - 3) * np.exp(-((x / b) ** c))
    # analytical solution
    top = (
        A
        * Nsat
        * b**3
        * (x / b) ** a
        * (c * (x / b) ** c - a + 3)
        * np.exp(-((x / b) ** c))
    )
    bot = x**4
    return -(top / bot)


def finite_difference(func: callable, x: float, h: float) -> float:
    """
    A building block to compute derivative using finite differences

    Parameters
    ----------
    func : callable
        Function to differentiate
    x : float
        Value(s) to evaluate derivative at
    h : float
        Step size for finite difference

    Returns
    -------
    dy : float
        Derivative at x
    """
    # return central difference
    return (func(x + h) - func(x - h)) / (2 * h)


def compute_derivative(
    func: callable,
    x: float,
    h_init: float,
    # For Ridders use parameters below:
    d: float = 2.0,
    eps: float = 1e-9,
    max_iters: int = 5,
) -> float:
    """
    Function to compute derivative

    Parameters
    ----------
    func : callable
        Function to differentiate
    x : float
        Value(s) to evaluate derivative at
    h_init : float
        Initial step size for finite difference
    d : float
        Factor by which to decrease h_init every iteration, default = 2.0
    eps : float
        Relative error, default at 1e-9
    max_iters : int
        Maximum number of iterations before exiting, default = 5

    Returns
    -------
    df : float
        Derivative at x
    """
    # local repo
    h = h_init
    d_mat = np.zeros((max_iters, max_iters))
    err_last = np.inf
    best_estimate = d_mat[0, 0]

    # fill 1st col
    for i in range(max_iters):
        d_mat[i, 0] = finite_difference(func, x, h)
        h /= d

    # fill the rest
    for j in range(1, max_iters):
        for i in range(max_iters - j):
            top = d ** (2 * j) * d_mat[i + 1, j - 1] - d_mat[i, j - 1]
            bot = d ** (2 * j) - 1
            d_mat[i, j] = top / bot

        # get err estimate
        err = abs((d_mat[0, j] - d_mat[0, j - 1]) / d_mat[0, j])
        if err < eps:
            return d_mat[0, j]
        if err > err_last * 1.2 and j > 1:
            return best_estimate
        # update
        best_estimate = d_mat[0, j]
        err_last = err

    return best_estimate


# DIFF - END
# ==================================================================================== #
# MAIN - START


def main():

    # Values from the hand-in
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    bounds = (0, 5)
    xmin, xmax = 10 ** (-4), 5
    N_generate = 10000
    xx = np.linspace(xmin, xmax, N_generate)

    def integrand(x: float, a: float = a, b: float = b, c: float = c):
        """Integrand function to replace original lambda fn call"""
        if x != 0.0:
            return 4.0 * np.pi * x**2 * (x / b) ** (a - 3) * np.exp(-((x / b) ** c))
        else:
            return 0.0

    integral, err = romberg_integrator(
        integrand, bounds, order=10, args=(a, b, c), err=True
    )

    # Normalisation
    A = 1.0 / integral
    with open("./output/a2q1_satellite_A.txt", "w") as f:
        f.write(f"{A:.12g}\n")

    def integrand_nsat(
        x: float,
        a: float = a,
        b: float = b,
        c: float = c,
        A: float = A,
        Nsat: int = Nsat,
    ):
        """Integrand with Nsat"""
        if x != 0.0:
            return 4.0 * np.pi * x**2 * n(x, A, Nsat, a, b, c)
        else:
            return 0.0

    integral_nsat = romberg_integrator(integrand_nsat, bounds, order=10, err=False)

    def p_of_x(x: float):
        """Get p(x)"""
        return integrand(x) / integral

    p_of_x_norm = p_of_x

    random_samples = sampler(
        p_of_x_norm, min_val=xmin, max_val=xmax, Nsamples=N_generate, args=()
    )

    edges = 10 ** np.linspace(np.log10(xmin), np.log10(xmax), 51)
    counts, _ = np.histogram(random_samples, bins=edges)
    bin_widths = np.diff(edges)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    hist_scaled = (counts / bin_widths) / (N_generate / Nsat)

    relative_radius = bin_centers
    analytical_function = np.array([Nsat * p_of_x(x) for x in relative_radius])

    # 1b plot gen
    fig1b, ax = plt.subplots()
    ax.stairs(
        hist_scaled, edges=edges, fill=True, alpha=0.6, label="Satellite galaxies"
    )
    plt.plot(relative_radius, analytical_function, "r-", label="Analytical solution")
    ax.set(
        xlim=(xmin, xmax),
        ylim=(1e-3, 1e3),  # you may or may not need to change ylim
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.savefig("./plots/a2q1_my_solution_1b.png", dpi=600)
    plt.close(fig1b)

    # Cumulative plot of the chosen galaxies (1c)
    chosen = choice(random_samples, size=100)
    chosen = sort_array(chosen)

    fig1c, ax = plt.subplots()
    ax.step(chosen, np.arange(100))
    ax.set(
        xscale="log",
        xlabel="Relative radius",
        ylabel="Cumulative number of galaxies",
        xlim=(xmin, xmax),
        ylim=(0, 100),
    )
    plt.savefig("./plots/a2q1_my_solution_1c.png", dpi=600)
    plt.close(fig1c)

    plot_rng_uniformity()

    x_to_eval = 1
    func_to_eval = lambda x: n(x, A, Nsat, a, b, c)

    dn_dx_numeric = compute_derivative(func_to_eval, x_to_eval, h_init=0.1)
    dn_dx_analytic = dn_dx(x_to_eval, A, Nsat, a, b, c)

    with open("./output/a2q1_satellite_deriv_analytic.txt", "w") as f:
        f.write(f"{dn_dx_analytic:.12g}\n")
    with open("./output/a2q1_satellite_deriv_numeric.txt", "w") as f:
        f.write(f"{dn_dx_numeric:.12g}\n")


# MAIN - END
# ==================================================================================== #

if __name__ == "__main__":
    main()
