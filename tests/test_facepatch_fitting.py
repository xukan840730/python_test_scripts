import math
import numpy as np
from face_prototype.face_patch import *


def test_misc(face_model, alphas1, v_targets):
    print("valid1:")
    for vertex_index in range(4):
        v_target = v_targets[vertex_index]
        str = ''
        for alpha_index in range(3):
            d2_c0 = calc_partial_derv_numerical_cwise(face_model, alphas1, alpha_index, vertex_index, v_target, 0)
            d2_c1 = calc_partial_derv_numerical_cwise(face_model, alphas1, alpha_index, vertex_index, v_target, 1)
            d2_c2 = calc_partial_derv_numerical_cwise(face_model, alphas1, alpha_index, vertex_index, v_target, 2)
            str += f' [{d2_c0, d2_c1, d2_c2}] '
        print(str)

    jacobian_comp = face_model.calc_f_jacobian_cwise(alphas1, v_targets)
    print(jacobian_comp)


def calc_r_norm(face_model, alphas, v_targets, w_fit, w_sparse):
    v = face_model.forward_pass(alphas)
    r1 = (v_targets - v) * w_fit
    r1_flatten = r1.flatten()
    r2 = alphas * w_sparse
    r_flatten = np.concatenate((r1_flatten, r2), axis=0)
    return np.linalg.norm(r_flatten)


def test_solve1(face_model, alphas1, v_targets):
    print('test_solve1:')
    alphas_best = face_model.solve_alphas_cwise(alphas1, v_targets, 100)
    print(f'alphas_best:{alphas_best}')
    r_best_norm = calc_r_norm(face_model, alphas_best, v_targets, 1.0, 0.0)
    print(f'r_best_norm:{r_best_norm}')

    eps = 0.0001
    alphas_perbs = []
    alphas_perbs.append(alphas_best - np.array([eps, 0, 0]))
    alphas_perbs.append(alphas_best + np.array([eps, 0, 0]))
    alphas_perbs.append(alphas_best - np.array([0, eps, 0]))
    alphas_perbs.append(alphas_best + np.array([0, eps, 0]))
    alphas_perbs.append(alphas_best - np.array([0, 0, eps]))
    alphas_perbs.append(alphas_best + np.array([0, 0, eps]))

    for i in range(len(alphas_perbs)):
        r_norm = calc_r_norm(face_model, alphas_perbs[i], v_targets, 1.0, 0.0)
        print(r_norm - r_best_norm)
        assert(r_norm > r_best_norm)


def test_solve2(face_model, alphas, v_targets, num_alphas):
    print('test_solve2:')
    w_fit = 1.0
    w_sparse = 1.0
    alphas_best = face_model.solve_alphas_cwise_2(alphas, v_targets, 100, w_fit, w_sparse)
    print(f'alphas_best:{alphas_best}')
    r_best_norm = calc_r_norm(face_model, alphas_best, v_targets, w_fit, w_sparse)
    print(f'r_best_norm:{r_best_norm}')

    eps = 0.0001
    alphas_perbs = []

    if num_alphas == 2:
        alphas_perbs.append(alphas_best - np.array([eps, 0]))
        alphas_perbs.append(alphas_best + np.array([eps, 0]))
        alphas_perbs.append(alphas_best - np.array([0, eps]))
        alphas_perbs.append(alphas_best + np.array([0, eps]))
    elif num_alphas == 3:
        alphas_perbs.append(alphas_best - np.array([eps, 0, 0]))
        alphas_perbs.append(alphas_best + np.array([eps, 0, 0]))
        alphas_perbs.append(alphas_best - np.array([0, eps, 0]))
        alphas_perbs.append(alphas_best + np.array([0, eps, 0]))
        alphas_perbs.append(alphas_best - np.array([0, 0, eps]))
        alphas_perbs.append(alphas_best + np.array([0, 0, eps]))
    else:
        assert False, 'not supported!'

    for i in range(len(alphas_perbs)):
        r_norm = calc_r_norm(face_model, alphas_perbs[i], v_targets, w_fit, w_sparse)
        print(r_norm - r_best_norm)
        assert(r_norm > r_best_norm)


def test_main_3():
    patch_neutral = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    patches_deltas = np.array(
        [# [[1, 0, 0], [0, 0.5, 0], [0, 0, 2]],  # deltas for the first vertex
         [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # deltas for the second vertex
         [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],  # deltas for the third vertex
         [[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]]],
    )

    face_model = FacePatch(patch_neutral, patches_deltas)

    alphas1 = np.array([0.1, 0.2, 0.3])
    shape1 = face_model.forward_pass(alphas1)
    print(shape1)

    v_targets = np.array([[1.3, 1.4, 1.5], [2.3, 2.4, 2.5], [3.1, 3.2, 3.3], [4.3, 4.2, 4.1]])

    # test_misc(face_model, alphas1, v_targets)
    test_solve1(face_model, alphas1, v_targets)
    test_solve2(face_model, alphas1, v_targets, 3)


def test_main_4():
    patch_neutral = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    patches_deltas = np.array(
        [  # [[1, 0, 0], [0, 0.5, 0], [0, 0, 2]],  # deltas for the first vertex
            [[1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1]],  # deltas for the second vertex
            [[0.5, 0, 0], [0, 0, 0.5]],  # deltas for the third vertex
            [[0.3, 0, 0], [0, 0, 0.3]]],
    )

    face_model = FacePatch(patch_neutral, patches_deltas)

    alphas1 = np.array([0.1, 0.3])
    shape1 = face_model.forward_pass(alphas1)
    print(shape1)

    v_targets = np.array([[1.3, 1.4, 1.5], [2.3, 2.4, 2.5], [3.1, 3.2, 3.3], [4.3, 4.2, 4.1]])

    # test_misc(face_model, alphas1, v_targets)
    # test_solve1(face_model, alphas1, v_targets)
    test_solve2(face_model, alphas1, v_targets, 2)


if __name__ == '__main__':
    test_main_4()
