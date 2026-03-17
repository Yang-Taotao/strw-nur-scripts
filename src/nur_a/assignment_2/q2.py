"""
Scripts for assignment 2 question 2
"""

import numpy as np

# Constants (mind the units!)

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


#### root finder ####


def root_finder(
    func: callable,  # add derivative if using Newton-Raphson
    bracket: tuple,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 100,
) -> tuple[float, float, float]:
    """
    Find a root of a function - Brent's method

    Parameters
    ----------
    func : callable
        Function to find root of
    bracket : tuple
        Bracket for which to find first secant
    atol : float, optional
        Absolute tolerance.
        The default is 1e-6
    rtol : float, optional
        Relative tolerance.
        The default is 1e-6
    max_iters: int, optional
        Maximum number of iterations.
        Teh default is 100

    Returns
    -------
    root : float
        Approximate root
    aerr : float
        Absolute error
    rerr : float
        Relative error
    """
    # TODO: Implement root finder (e.g. bisection, false-position, Newton-Raphson)
    return 0.0, 0.0, 0.0


def main():

    # Initial bracket
    bracket = (1, 1e7)

    root, aerr, rerr = 0.0, 0.0, 0.0  # replace with your root finder

    with open("./output/a2q2_equilibrium_temp_simple.txt", "w") as f:
        f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e}")
    #### 2b ####

    # Initial bracket
    bracket = (1, 1e15)

    for nH in [1e-4, 1, 1e4]:

        root, aerr, rerr = 0.0, 0.0, 0.0  # replace with your root finder
        if nH == 1e-4:
            with open("./output/a2q2_equilibrium_low_density.txt", "w") as f:
                f.write(f"{root:.12g}")
        elif nH == 1:
            with open("./output/a2q2_equilibrium_mid_density.txt", "w") as f:
                f.write(f"{root:.12g}")
        elif nH == 1e4:
            with open("./output/a2q2_equilibrium_high_density.txt", "w") as f:
                f.write(f"{root:.12g}")


if __name__ == "__main__":
    main()
