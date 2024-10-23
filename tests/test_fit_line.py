import numpy as np


def fit_line_1(l_x, l_y):
    xs = np.array(l_x)
    ys = np.array(l_y)
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)

    num = 0.0
    den = 0.0
    for i in range(len(xs)):
        num += (xs[i] - x_mean) * (ys[i] - y_mean)
        den += (xs[i] - x_mean) ** 2

    m = num / den
    b = y_mean - m * x_mean
    return m, b


def fit_line_2(l_x, l_y):
    assert len(l_x) == len(l_y)

    N = len(l_x)
    sum_x = sum(l_x)
    sum_y = sum(l_y)
    sum_xy = sum(l_x[i] * l_y[i] for i in range(N))
    sum_x2 = sum(l_x[i]**2 for i in range(N))

    m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)
    b = (sum_y - m * sum_x) / N
    return m, b


if __name__ == '__main__':
    # l_x = [1, 2, 3, 4, 5]
    # l_y = [2, 3, 4, 5, 6]

    l_x = np.array([1, 2, 3, 4, 5])
    l_y = np.array([1.2, 1.9, 3.0, 3.9, 5.1])

    m1, b1 = fit_line_1(l_x, l_y)
    print(f'm1: {m1}, b1: {b1}')

    m2, b2 = fit_line_2(l_x, l_y)
    print(f'm2: {m2}, b2: {b2}')

