import numpy as np


def to_shape(neutral, diff, alphas):
    num_alphas = alphas.shape[0]
    a = np.dot(alphas, diff)
    return neutral + a


def test_main():
    patch_target = np.array([[1.3, 1.4, 1.5], [2.3, 2.4, 2.5]])
    num_patches = patch_target.shape[0]

    patch_neutral = np.array([[1, 1, 1], [2, 2, 2]])
    patches_d = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]]])
    num_alphas = patches_d.shape[1]

    alphas1 = np.array([0.1, 0.2, 0.3])
    x1 = to_shape(patch_neutral[0], patches_d[0], alphas1)
    print(x1)

    x2 = to_shape(patch_neutral[1], patches_d[1], alphas1)
    print(x2)


if __name__ == '__main__':
    test_main()