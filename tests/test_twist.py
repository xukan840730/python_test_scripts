import math
import numpy as np


def test_twist_to_rotation_matrix():
    wx = -0.1
    wy = -1.0
    wz = 0.8
    # wx = 0.707
    # wy = -0.707
    # wz = 0
    w = np.array([wx, wy, wz])
    theta = np.linalg.norm(w)
    w_hat = np.array(
        [[0, -wz, wy],
         [wz, 0, -wx],
         [-wy, wx, 0]])

    print('exp_test')
    # exp1 = np.identity(3) + w_hat * math.sin(theta) + w_hat*w_hat*(1 - math.cos(theta))
    # print(exp1)
    exp2 = np.identity(3) + w_hat * math.sin(theta) + np.matmul(w_hat, w_hat) * (1 - math.cos(theta))
    print(exp2)

    exp3 = np.identity(3) + \
           w_hat + \
           (w_hat@w_hat) / 2 + \
           (w_hat@w_hat@w_hat) / 6 + \
           (w_hat@w_hat@w_hat@w_hat) / 24 + \
           (w_hat @ w_hat @ w_hat @ w_hat@w_hat) / 120
    print(exp3)
    print(np.linalg.norm(exp3[0]))
    print(np.linalg.norm(exp3[1]))
    print(np.linalg.norm(exp3[2]))
    # exp4 = np.identity(3) + w_hat + (w_hat*w_hat) / 2 + (w_hat*w_hat*w_hat) / 6 + (w_hat*w_hat*w_hat*w_hat) / 24
    # print(exp4)

    exp5 = np.identity(3)
    m = np.identity(3)
    b = 1
    for i in range(100):
        m = m @ w_hat
        b *= (i + 1.0)
        exp5 += m / b
    print(exp5)
    print(np.linalg.norm(exp5[0]))
    print(np.linalg.norm(exp5[1]))
    print(np.linalg.norm(exp5[2]))


if __name__ == '__main__':
    test_twist_to_rotation_matrix()