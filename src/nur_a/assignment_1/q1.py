"""
Scripts for assignment 1 question 1
"""

import numpy as np


def Poisson(k: np.int32, lmbda: np.float32) -> np.float32:
    """Calculate the Poisson probability for k occurrences with mean lmbda.
    Parameters:
        k (np.int32): The number of occurrences.
        lmbda (np.float32): The mean number of occurrences.
    Returns:
        np.float32: The probability of observing k occurrences given the mean lmbda.
    """
    # we start with the gen form of posson distro P
    # top = lmbda**k * np.exp(-lmbda)
    # bot = np.prod(np.arange(1, k, 1))
    # P = top / bottom

    # we then rewrite P with log space to prevent over/underflow
    # ln(P) = k ln(lmbda) - lmbda - ln(k!)
    # with ln(k!) = sum(ln(i) for i in range(1, k+1))
    # P <-> ln(P) through np.exp()

    # break if k and lmbda doesn't match wanted dtype
    if k.dtype != "int32" or lmbda.dtype != "float32":
        raise TypeError(
            f"No matching dtype with k.dtype {k.dtype}, lmbda.dtype {lmbda.dtype}"
        )

    # local dtype enforcement
    k = np.int32(k)
    lmbda = np.float32(lmbda)

    # special case if k = 0 -> k! = 1
    if k == np.int32(0):
        result = np.exp(-lmbda)
    # break if k is neg
    elif k < np.int32(0):
        raise ValueError(f"Invalid k at negative value, k={k}.")
    # log space rewrite P -> ln(P)
    else:
        log_factorial = sum(np.log(np.arange(1, k, 1)))
        log_distro = k * np.log(lmbda) - log_factorial
        result = np.exp(log_distro)

    return result


def main() -> None:
    # (lambda, k) pairs:
    values = [
        (np.float32(1.0), np.int32(0)),
        (np.float32(5.0), np.int32(10)),
        (np.float32(3.0), np.int32(21)),
        (np.float32(2.6), np.int32(40)),
        (np.float32(100.0), np.int32(5)),
        (np.float32(101.0), np.int32(200)),
    ]
    with open("./output/poisson_output.txt", "w") as file:
        for i, (lmbda, k) in enumerate(values):
            P = Poisson(k, lmbda)
            if i < len(values) - 1:
                file.write(f"{lmbda:.1f} & {k} & {P:.6e} \\\\ \\hline \n")
            else:
                file.write(f"{lmbda:.1f} & {k} & {P:.6e} \n")


if __name__ == "__main__":
    main()
