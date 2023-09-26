import numpy as np


# single face patch
class FacePatch:
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

    def calc_partial_derv(self, vertex_index, alphas, alpha_index, v_target):
        vertex_deltas = self._deltas[vertex_index]
        f = v_target - (self._neutral[vertex_index] + np.dot(alphas, vertex_deltas))
        t3 = np.dot(f, vertex_deltas[alpha_index])
        return -2 * t3


def calc_partial_derv_numerical(in_model, in_alphas, alpha_index, vertex_index, v_target):
    eps = 0.0001
    perb = np.zeros(in_alphas.shape)
    perb[alpha_index] = eps
    f1 = v_target - in_model.forward_pass(in_alphas - perb)[vertex_index]
    f2 = v_target - in_model.forward_pass(in_alphas + perb)[vertex_index]
    y1 = np.dot(f1, f1)
    y2 = np.dot(f2, f2)
    derv = (y2 - y1) / (2 * eps)
    return derv


def test_main():
    # patch_target = np.array([[1.3, 1.4, 1.5], [2.3, 2.4, 2.5]])
    # num_patches = patch_target.shape[0]

    patch_neutral = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    patches_deltas = np.array(
        [[[1, 0, 0], [0, 0.5, 0], [0, 0, 2]],  # deltas for the first vertex
         [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # deltas for the second vertex
         [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],  # deltas for the third vertex
         [[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]]],
    )

    face_model = FacePatch(patch_neutral, patches_deltas)

    alphas1 = np.array([0.1, 0.2, 0.3])
    shape1 = face_model.forward_pass(alphas1)
    print(shape1)

    v1_target = [1.3, 1.4, 1.5]

    for alpha_index in range(3):
        for vertex_index in range(4):
            d1 = face_model.calc_partial_derv(vertex_index, alphas1, alpha_index, v1_target)
            d2 = calc_partial_derv_numerical(face_model, alphas1, alpha_index, vertex_index, v1_target)
            print(f'{d1}, {d2}')


if __name__ == '__main__':
    test_main()