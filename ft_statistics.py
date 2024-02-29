import numpy as np


def apply_count(column):
    count = 0
    for row in column:
        count += 1
    return count


def apply_min(column):
    min = None
    for row in column:
        if min is None or row < min:
            min = row
    return min


def apply_max(column):
    max = None
    for row in column:
        if max is None or row > max:
            max = row
    return max


def apply_mean(column):
    count = apply_count(column)
    return column.sum() / count


def apply_std(column):
    square_diff = 0
    count = 0
    mean = apply_mean(column)
    for row in column:
        square_diff += np.square(row - mean)
        count += 1
    return np.sqrt(square_diff / (count - 1))


def apply_first_quartile(column):
    sorted_column = column.sort()
    count = apply_count(column)
    index = (count - 1) / 4
    if index.is_integer():
        return sorted_column[int(index)]

    lower_index = int(index)
    upper_index = lower_index + 1
    interpolation_factor = index - lower_index
    return (1 - interpolation_factor) * sorted_column[
        lower_index
    ] + interpolation_factor * sorted_column[upper_index]


def apply_median(column):
    sorted_column = column.sort()
    count = apply_count(column)
    index = count // 2
    if count % 2 == 0:
        return (sorted_column[index - 1] + sorted_column[index]) / 2
    return sorted_column[index]


def apply_last_quartile(column):
    sorted_column = column.sort()
    count = apply_count(column)
    index = 3 * (count - 1) / 4
    if index.is_integer():
        return sorted_column[int(index)]

    lower_index = int(index)
    upper_index = lower_index + 1
    interpolation_factor = index - lower_index
    return (1 - interpolation_factor) * sorted_column[
        lower_index
    ] + interpolation_factor * sorted_column[upper_index]
