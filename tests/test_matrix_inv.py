import numpy as np


def test_func():
    vertices = np.array([[1.0, 0, 0],
                        [0, 0.0, 2.0],
                        [0.0, 0.0, 3.0],
                        [-1.0, -1.0, -1.0]])

    M = np.array([[vertices[1, 0] - vertices[0, 0], vertices[2, 0] - vertices[0, 0], vertices[3, 0] - vertices[0, 0]],
                  [vertices[1, 1] - vertices[0, 1], vertices[2, 1] - vertices[0, 1], vertices[3, 1] - vertices[0, 1]],
                  [vertices[1, 2] - vertices[0, 2], vertices[2, 2] - vertices[0, 2], vertices[3, 2] - vertices[0, 2]]])
    print(M)

    M_inv = np.linalg.inv(M)
    print(M_inv)


if __name__ == '__main__':
    test_func()
