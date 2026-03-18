"""
Scripts for assignment 2 question 2
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================================================================================== #
# EQFN - START

PSI = 0.929
TC = 1e4  # K
Z = 0.015
K = 1.38e-16  # erg K^(-1)
AB = 2e-13  # cm^(3) s^(-1)
A = 5e-10  # erg
XI = 1e-15  # s^(-1)


def equilibrium1(t: float, z: float = Z, tc: float = TC, psi: float = PSI) -> float:
    """Equilibrium fn 1 for photonization heating and recombination - simple case"""
    # simple case
    # fn = Gamma - Lambda
    return (psi * tc * K) - ((0.684 - 0.0416 * np.log(t / (1e4 * z * z))) * t * K)


def equilibrium2(
    t: float,
    nh: float,
    z: float = Z,
    tc: float = TC,
    psi: float = PSI,
    a: float = A,
    xi: float = XI,
    ab: float = AB,
) -> float:
    """Equilibrium fn 2 for photonization heating and recombination - complex case"""
    # full eq with other factors
    # fn = (Gamma_pe + Gamma_cr + Gamma_mhd) - (Lambda_rr + Lambda_ff)
    return (
        (
            # Gamma_pe
            psi * tc
            # Lambda_rr
            - (0.684 - 0.0416 * np.log(t / (1e4 * z * z))) * t
            # Lambda_ff
            - 0.54 * (t / 1e4) ** 0.37 * t
        )
        * K
        * nh
        * ab
        # Gamma_cr
        + a * xi
        # Gamma_mhd
        + 8.9e-26 * (t / 1e4)
    )


# EQFN - END
# ==================================================================================== #
# ROOT - START


def root_finder(
    func: callable,
    bracket: tuple[float, float],
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 1000,
) -> tuple[float, float, float]:
    """
    Find a root of a function - Brent's method

    Parameters
    ----------
    func : callable
        Function fn to find root of
    bracket : tuple
        Bracket for which to find root
    atol : float, optional
        Absolute tolerance.
        The default is 1e-6
    rtol : float, optional
        Relative tolerance.
        The default is 1e-6
    max_iters: int, optional
        Maximum number of iterations.
        The default is 100

    Returns
    -------
    root : float
        Approximate root
    aerr : float
        Absolute error - final bracket width
    rerr : float
        Relative error - final width / |fb|
    """
    # local repo
    a, b = bracket
    fa, fb = func(a), func(b)

    # ensure at least one root is present in bracket
    # ideally only one root present is the best case
    if fa * fb >= 0.0:
        raise ValueError(f"Cannot guarantee n_root >= 1 with set bracket {bracket}.")

    # set b as the best estimate between a and b
    # must have abs(fa) > abs(fb)
    # no swap needed if equal scale
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    # set initial c as a
    # at next iter c is current b
    c = a
    niter = max_iters

    # init new point and diff size
    d = 0.5 * (b - a)
    e = d

    # make current state
    state = (a, b, c, d, e, niter)

    def check_bracket(state: tuple) -> tuple:
        """Check if the current [c, b] bracket is still valid"""
        # unpack and get val
        a, b, c, d, e, niter = state
        fb, fc = func(b), func(c)
        # check bracket
        if fb * fc > 0.0:
            e = b - a
            # fallback to c = a to ensure valid bracket
            state = (a, b, a, d, e, niter)

        return state

    def check_best_estimate(state: tuple) -> tuple:
        """Best estimation should be x=b"""
        a, b, c, d, e, niter = state
        fb, fc = func(b), func(c)

        # set b as best estimate
        # if |fb| > |fc| -> c is better estimate now
        if abs(fb) > abs(fc):
            # set a_new -> b, b_new -> c, c_new -> a
            state = (b, c, a, d, e, niter)

        return state

    def find_new_point(state: tuple) -> tuple:
        """Calculate d from current state"""
        a, b, c, d, e, niter = state
        fa, fb, fc = func(a), func(b), func(c)

        # method select
        if c == a:
            # secant method
            d_new = b - fb * (b - a) / (fb - fa)
        else:
            # quadratic
            r, s, t = fb / fc, fb / fa, fa / fc
            p = s * (t * (r - t) * (c - b) - (1.0 - r) * (b - a))
            q = (t - 1.0) * (r - 1.0) * (s - 1.0)
            d_new = b + p / q

        state = (a, b, c, d_new, e, niter)

        return state

    def check_d(state: tuple) -> tuple:
        """Check if d satisfy the conditions"""
        a, b, c, d, e, niter = state
        low = 0.25 * (3 * a + b)
        high = b
        h = abs(d - b)
        h_last = abs(e)
        # bisection conditions check
        if (low <= d <= high) and (h < 0.5 * h_last):
            # no bisection
            d_new = d
            e_new = d - b
        else:
            # use bisection
            d_new = 0.5 * (a + b)
            e_new = b - a

        return (a, b, c, d_new, e_new, niter)

    def are_we_there_yet(state: tuple) -> tuple:
        """Check if a root is found, otherwise update state"""
        a, b, c, d, e, niter = state
        fb, fd = func(b), func(d)

        # exit if new point is root
        if fd == 0.0:
            return state

        # update bracket
        if fb * fd < 0.0:
            a_new, b_new = b, d
        else:
            a_new, b_new = a, d

        # enforce b as the best estimate
        fa_new, fb_new = func(a_new), func(b_new)
        # if |fb| > |fa| -> a is better estimate -> swap
        if abs(fb_new) > abs(fa_new):
            a_new, b_new = b_new, a_new

        # update c_new = b
        c_new = b

        # update niter counter
        niter -= 1

        # rebuild state
        state = (a_new, b_new, c_new, d, e, niter)

        return state

    def check_convergence(state: tuple) -> tuple:
        """Recursion stop consition checker"""
        a, b, c, d, e, niter = state
        fb = func(b)

        # get current tolerance
        tol = atol + rtol * abs(b)
        aerr = abs(b - a)
        rerr = aerr
        flag = False
        root = None

        # check convergence
        if (aerr < tol) or (niter == 0) or (fb == 0.0):
            # yes convergence
            flag = True
            root = b
            if b != 0.0:
                rerr = aerr / abs(b)

        # build output for main return
        result = (flag, niter, root, aerr, rerr)

        return result

    def evolve(state: tuple) -> tuple:
        """Recursivly do Brent's method to get the final state"""
        flag, niter, root, aerr, rerr = check_convergence(state)

        # exit if convergence achieved
        if flag == True:
            return root, aerr, rerr, max_iters - niter

        # main pipeline
        # validity check
        state = check_bracket(state)
        state = check_best_estimate(state)
        # find new point d
        state = find_new_point(state)
        # check if d is valid
        state = check_d(state)
        # evaluate and check
        state = are_we_there_yet(state)

        # recursive
        return evolve(state)

    return evolve(state)


# ROOT - END
# ==================================================================================== #
# MAIN - START


def main():

    # fig init
    fig, ax = plt.subplots()

    #### 2a ####
    # Initial bracket
    bracket_2a = (1, 1e7)

    root, aerr, rerr, niter = root_finder(func=equilibrium1, bracket=bracket_2a)

    with open("./output/a2q2_equilibrium_temp_simple.txt", "w") as f:
        f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e} & {niter}")

    #### 2b ####
    # Initial bracket
    bracket_2b = (1, 1e15)
    # density
    density = [1e-4, 1, 1e4]
    # temperature array
    t_ary = np.logspace(1, 15, 10000)

    for nH in density:

        # make lambda fn for nH assignment
        temp_func = lambda t, nh=nH: equilibrium2(t, nh)

        # results
        root, aerr, rerr, niter = root_finder(func=temp_func, bracket=bracket_2b)

        # write
        if nH == 1e-4:
            with open("./output/a2q2_equilibrium_low_density.txt", "w") as f:
                f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e} & {niter}")
        elif nH == 1:
            with open("./output/a2q2_equilibrium_mid_density.txt", "w") as f:
                f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e} & {niter}")
        elif nH == 1e4:
            with open("./output/a2q2_equilibrium_high_density.txt", "w") as f:
                f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e} & {niter}")

        # get f(t)
        f_t = temp_func(t_ary)

        # plot gen
        ax.loglog(t_ary, f_t, label=f"$n_H = {nH:.0e}$ cm$^{{-3}}$")
        ax.axvline(root, color="black", alpha=0.6)

    ax.set_xlabel("Temperature $T$ K")
    ax.set_ylabel("Equalibrium function evaluation")
    ax.legend()
    plt.savefig("./plots/a2q2_equilibrium_function.png", dpi=600)
    plt.close(fig)


# MAIN - END
# ==================================================================================== #

if __name__ == "__main__":
    main()
