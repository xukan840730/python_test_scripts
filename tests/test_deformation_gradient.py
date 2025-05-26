import numpy as np


# this func tests ComputeH on <dynamics deformable> page 219
# Xi is natural coordinate (u, v, w)
def compute_dN_dXi(Xi):
    assert -1.0 <= Xi[0] <= 1.0  # u
    assert -1.0 <= Xi[1] <= 1.0  # v
    assert -1.0 <= Xi[2] <= 1.0  # w

    b0Minus = 1.0 - Xi[0]
    b0Plus = 1.0 + Xi[0]
    b1Minus = 1.0 - Xi[1]
    b1Plus = 1.0 + Xi[1]
    b2Minus = 1.0 - Xi[2]
    b2Plus = 1.0 + Xi[2]

    H = np.zeros((8, 3))

    # (1 - u)(1 - v)(1 - w)
    H[0, 0] = -b1Minus * b2Minus
    H[0, 1] = -b0Minus * b2Minus
    H[0, 2] = -b0Minus * b1Minus

    # (1 + u)(1 - v)(1 - w)
    H[1, 0] = b1Minus * b2Minus
    H[1, 1] = -b0Plus * b2Minus
    H[1, 2] = -b0Plus * b1Minus

    # (1 - u)(1 + v)(1 - w)
    H[2, 0] = -b1Plus * b2Minus
    H[2, 1] = b0Minus * b2Minus
    H[2, 2] = -b0Minus * b1Plus

    # (1 + u)(1 + v)(1 - w)
    H[3, 0] = b1Plus * b2Minus
    H[3, 1] = b0Plus * b2Minus
    H[3, 2] = -b0Plus * b1Plus

    # (1 - u)(1 - v)(1 + w)
    H[4, 0] = -b1Minus * b2Plus
    H[4, 1] = -b0Minus * b2Plus
    H[4, 2] = b0Minus * b1Minus

    # (1 + u)(1 - v)(1 + w)
    H[5, 0] = b1Minus * b2Plus
    H[5, 1] = -b0Plus * b2Plus
    H[5, 2] = b0Plus * b1Minus

    # (1 - u)(1 + v)(1 + w)
    H[6, 0] = -b1Plus * b2Plus
    H[6, 1] = b0Minus * b2Plus
    H[6, 2] = b0Minus * b1Plus

    # (1 + u)(1 + v)(1 + w)
    H[7, 0] = b1Plus * b2Plus
    H[7, 1] = b0Plus * b2Plus
    H[7, 2] = b0Plus * b1Plus

    return H * 0.125


def test_compute_dN_dXi():
    H0 = compute_dN_dXi(np.array([-1.0, -1.0, -1.0]))
    print(H0)

    H1 = compute_dN_dXi(np.array([1.0, -1.0, -1.0]))
    print(H1)

    H2 = compute_dN_dXi(np.array([-1.0, 1.0, -1.0]))
    print(H2)

    H3 = compute_dN_dXi(np.array([1.0, 1.0, -1.0]))
    print(H3)

    H4 = compute_dN_dXi(np.array([-1.0, -1.0, 1.0]))
    print(H4)

    H5 = compute_dN_dXi(np.array([1.0, -1.0, 1.0]))
    print(H5)

    H6 = compute_dN_dXi(np.array([-1.0, 1.0, 1.0]))
    print(H6)

    H7 = compute_dN_dXi(np.array([1.0, 1.0, 1.0]))
    print(H7)


if __name__ == '__main__':
    test_compute_dN_dXi()
