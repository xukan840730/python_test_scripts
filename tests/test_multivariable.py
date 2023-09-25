import numpy as np


class FacePatch:
    # single face patch
    def __init__(self, in_neutral, in_deltas):
        assert(in_neutral.shape[0] == in_deltas.shape[0])
        self._neutral = in_neutral  # neutral shape
        self._deltas = in_deltas  # delta shapes

    def get_num_vertices(self):
        return self._neutral.shape[0]

    def get_num_delta_shapes(self):
        return self._deltas.shape[0]

    def forward_pass(self, alphas):  # alpha: deltas weights
        assert(alphas.shape[0] == self._deltas.shape[1])
        t = np.dot(alphas, self._deltas)
        return self._neutral + t


def to_shape(neutral, diff, alphas):
    num_alphas = alphas.shape[0]
    a = np.dot(alphas, diff)
    return neutral + a


def test_main():
    # patch_target = np.array([[1.3, 1.4, 1.5], [2.3, 2.4, 2.5]])
    # num_patches = patch_target.shape[0]

    patch_neutral = np.array([[1, 1, 1], [2, 2, 2]])
    patches_deltas = np.array(
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # deltas for the first vertex
         [[2, 0, 0], [0, 2, 0], [0, 0, 2]]]  # deltas for the second vertex
    )

    face_model = FacePatch(patch_neutral, patches_deltas)

    alphas1 = np.array([0.1, 0.2, 0.3])
    shape1 = face_model.forward_pass(alphas1)
    print(shape1)


if __name__ == '__main__':
    test_main()