"""
Scripts for tutorial session 6
"""

import numpy as np


def selection_sort(data: np.ndarray):
    """selection sort"""
    temp = data.copy()

    for i in range(0, len(temp) - 1):
        i_min = i

        for j in range(i_min + 1, len(temp)):
            if temp[j] < temp[i_min]:
                i_min = j

        if i_min != i:
            temp[i], temp[i_min] = temp[i_min], temp[i]

    return temp


def quick_sort(data: np.ndarray):
    """quicksort"""
    temp = data.copy()

    idx_mid = (len(temp) - 1) // 2

    temp_ary = np.array([temp[0], temp[idx_mid], temp[-1]])
    temp_ary = selection_sort(temp_ary)
    pivot = temp_ary[1]

    temp[0], temp[idx_mid], temp[-1] = temp_ary[0], temp_ary[1], temp_ary[-1]

    i, j = 0, len(temp) - 1
    count = 0

    print(i, j)
    print(temp)

    while i < j and count <= 30:
        while temp[i] < pivot:
            i += 1
        while temp[j] > pivot:
            j -= 1

        temp[i], temp[j] = temp[j], temp[i]

        count += 1
        print(temp)

    return temp


if __name__ == "__main__":
    data = np.array([2, 1, 8, 5, 4, 3])
    # selection_sort(data)
    quick_sort(data)
    # [2,1,8,5,4,3]
    # [2,1,3,5,4,8] -> pivot 3
