"""
Scripts for assignment 2 question 1
"""

# imports
import numpy as np
import matplotlib.pyplot as plt

# RNG - START
####################
## RNGS RNGS RNGS ##
####################

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

    def __init__(self, seed: int = 42):
        """Seed assignment init"""
        # avoid zero seed for xor
        if seed == 0:
            seed = 42

        seed &= MASK_64
        self.state_xor = seed
        self.state_mwc = seed

    def nxt_iter(self):
        """Get to next iter, given current iter results, return rng_val"""
        nxt_state_xor, nxt_state_mwc, rng_val = rng_gen(self.state_xor, self.state_mwc)
        self.state_xor = nxt_state_xor
        self.state_mwc = nxt_state_mwc
        return rng_val

    def rand(self):
        """Return normalized rng_val similar to np.random.rand()"""
        return self.nxt_iter() / NUM_2_64


rng = NURARNG()


# RNG - END


# NDEN - START
####################
## NDEN NDEN NDEN ##
####################


def n(
    x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> float | np.ndarray:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : float | ndarray
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
    float | ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    return A * Nsat * (x / b) ** (a - 3) * np.exp(-((x / b) ** c))


# NDEN - END

# INTG - START
####################
## INTG INTG INTG ##
####################


# Below we provide a template for romberg integration
# You can implement this
# or use another integration method based on some form of Richardson extrapolation
def romberg_integrator(
    func: callable, bounds: tuple, order: int = 5, err: bool = False, args: tuple = ()
) -> float | tuple[float, float]:
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

# SAMP - START
####################
## SAMP SAMP SAMP ##
####################


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
    delta = 0.01 * (max_val - min_val)

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

# SORT - START
####################
## SORT SORT SORT ##
####################


def sort_array(
    arr: np.ndarray,
    inplace: bool = False,
) -> np.ndarray:
    """
    Sort a 1D array using a sorting algorithm of your choice

    Parameters
    ----------
    arr : ndarray
        Input array to be sorted
    inplace : bool, optional
        If True, sort the array in-place
        If False, return a sorted copy

    Returns
    -------
    sorted_arr : ndarray
        Sorted array (same shape as arr)

    """
    if inplace:
        sorted_arr = arr
    else:
        sorted_arr = arr.copy()

    # TODO: sort sorted_arr in-place here

    return sorted_arr


def choice(arr: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Choose given number of random elements from an array, without replacement

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
    # TODO: Implement your choice function here, e.g. by using Fisher-Yates shuffling
    return arr[:size].copy()


# SORT - END

# DIFF - START
####################
## DIFF DIFF DIFF ##
####################


def dn_dx(
    x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> float | np.ndarray:
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
    float | ndarray
        Same type and shape as x. Derivative of number density of
        satellite galaxies at given radius x.
    """
    # TODO: Write the analytical derivative of n(x) here
    return 0.0


def finite_difference(
    function: callable, x: float | np.ndarray, h: float
) -> float | np.ndarray:
    """
    A building block to compute derivative using finite differences

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h : float
        Step size for finite difference

    Returns
    -------
    dy : float | ndarray
        Derivative at x
    """
    # TODO: Implement finite difference method
    return 0.0


def compute_derivative(
    function: callable,
    x: float | np.ndarray,
    h_init: float,
    # For Ridders use parameters below:
    # d: float, # Factor by which to decrease h_init every iteration
    # eps: float, # Relative error
    # max_iters: int = 10, 3 Maximum number of iterations before exiting
) -> float | np.ndarray:
    """
    Function to compute derivative

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h_init : float
        Initial step size for finite difference

    Returns
    -------
    df : float | ndarray
        Derivative at x
    """
    # TODO: Implement derivative
    return 0.0


# DIFF - END

# MAIN - START
####################
## MAIN MAIN MAIN ##
####################


def main():

    # Values from the hand-in
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    bounds = (0, 5)
    xmin, xmax = 10**-4, 5
    N_generate = 10000
    xx = np.linspace(xmin, xmax, N_generate)

    integrand = lambda x, a, b, c: 0.0  # insert the correct function
    integral, err = romberg_integrator(
        integrand, bounds, order=2, args=(a, b, c), err=True
    )

    # Normalisation
    A = 1.0  # to be computed
    with open("./output/a2q1_satellite_A.txt", "w") as f:
        f.write(f"{A:.12g}\n")
    integrand = lambda x, a, b, c: 0.0  # replace by the correct function
    integrated_Nsat = (
        0.0  # replace by the correct integral, e.g. by calling your integrator
    )

    p_of_x = (
        lambda x: 0.0
    )  # replace by the normalised distribution of satellite galaxies as a function of x

    # Numerically determine maximum to normalize p(x) for sampling
    pmax = 0.0  # replace by taking the maximum value of p_of_x

    p_of_x_norm = lambda x: 0.0  # replace by the normalised distribution
    random_samples = np.zeros(
        N_generate
    )  # replace by your sampler(p_of_x_norm, min=xmin, max=xmax, Nsamples=N_generate, args=())

    edges = 10 ** np.linspace(np.log10(xmin), np.log10(xmax), 21)

    hist = np.histogram(
        xmin + np.sort(np.random.rand(N_generate)) * (xmax - xmin), bins=edges
    )[
        0
    ]  # replace!
    hist_scaled = (
        1e-3 * hist
    )  # replace; this is NOT what you should be plotting, this is just a random example to get a plot with reasonable y values (think about how you *should* scale hist)

    fig = plt.figure()
    relative_radius = edges.copy()  # replace!
    analytical_function = edges.copy()  # replace

    fig1b, ax = plt.subplots()
    ax.stairs(
        hist_scaled, edges=edges, fill=True, label="Satellite galaxies"
    )  # just an example line, correct this!
    plt.plot(
        relative_radius, analytical_function, "r-", label="Analytical solution"
    )  # correct this according to the exercise!
    ax.set(
        xlim=(xmin, xmax),
        ylim=(10 ** (-3), 10),  # you may or may not need to change ylim
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.savefig("./plots/a2q1_my_solution_1b.png", dpi=600)

    # Cumulative plot of the chosen galaxies (1c)
    chosen = xmin + np.sort(np.random.rand(Nsat)) * (xmax - xmin)  # replace!
    fig1c, ax = plt.subplots()
    ax.plot(chosen, np.arange(100))
    ax.set(
        xscale="log",
        xlabel="Relative radius",
        ylabel="Cumulative number of galaxies",
        xlim=(xmin, xmax),
        ylim=(0, 100),
    )
    plt.savefig("./plots/a2q1_my_solution_1c.png", dpi=600)

    x_to_eval = 1
    func_to_eval = lambda x: n(x, A, Nsat, a, b, c)
    dn_dx_numeric = 0.0  # replace by your derivative, e.g. compute_derivative(func_to_eval, x_to_eval, h_init=0.1)
    dn_dx_analytic = dn_dx(x_to_eval, A, Nsat, a, b, c)
    with open("./output/a2q1_satellite_deriv_analytic.txt", "w") as f:
        f.write(f"{dn_dx_analytic:.12g}\n")

    with open("./output/a2q1_satellite_deriv_numeric.txt", "w") as f:
        f.write(f"{dn_dx_numeric:.12g}\n")


# MAIN - END

####################
## EXEC EXEC EXEC ##
####################

if __name__ == "__main__":
    main()
