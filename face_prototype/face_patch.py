import numpy as np
import scipy.linalg


# single face patch
class FacePatch:
    def __init__(self, in_neutral, in_deltas):
        assert(in_neutral.shape[0] == in_deltas.shape[0])
        self._neutral = in_neutral  # neutral shape
        self._deltas = in_deltas  # delta shapes

    def get_num_vertices(self):
        return self._neutral.shape[0]

    def get_num_delta_shapes(self):
        return self._deltas.shape[1]

    def forward_pass(self, alphas):  # alpha: deltas weights
        assert(alphas.shape[0] == self._deltas.shape[1])
        t = np.dot(alphas, self._deltas)
        return self._neutral + t

    def calc_f_partial_derv_cwise(self, vertex_index, alphas, alpha_index, v_target):
        vertex_deltas = self._deltas[vertex_index]
        d = vertex_deltas[:, alpha_index]
        return -d

    def calc_f_jacobian_cwise(self, in_alphas, v_targets):
        num_vertices = self.get_num_vertices()
        assert (v_targets.shape[0] == num_vertices)
        num_delta_shapes = self.get_num_delta_shapes()

        jacobian = np.zeros((num_vertices * 3, num_delta_shapes))  # 3d vertex
        for vertex_index in range(num_vertices):
            v_target = v_targets[vertex_index]
            for alpha_index in range(num_delta_shapes):
                derv_comp = self.calc_f_partial_derv_cwise(vertex_index, in_alphas, alpha_index, v_target)
                fill_index = vertex_index * 3
                jacobian[fill_index:fill_index + 3, alpha_index] = derv_comp

        return jacobian

    # version 1: alphas can be arbitrary. It's probably useless in real case.
    def solve_alphas_cwise(self, in_alphas, v_targets, in_max_iter):
        alphas = np.copy(in_alphas)
        # print(alphas)

        for i in range(in_max_iter):
            jacobian = self.calc_f_jacobian_cwise(alphas, v_targets)

            # validation:
            for vertex_index in range(self.get_num_vertices()):
                v_target = v_targets[vertex_index]
                for alpha_index in range(self.get_num_delta_shapes()):
                    d_c0 = calc_partial_derv_numerical_cwise(self, alphas, alpha_index, vertex_index, v_target, 0)
                    d_c1 = calc_partial_derv_numerical_cwise(self, alphas, alpha_index, vertex_index, v_target, 1)
                    d_c2 = calc_partial_derv_numerical_cwise(self, alphas, alpha_index, vertex_index, v_target, 2)
                    assert (abs(jacobian[vertex_index * 3 + 0, alpha_index] - d_c0) < 0.0001)
                    assert (abs(jacobian[vertex_index * 3 + 1, alpha_index] - d_c1) < 0.0001)
                    assert (abs(jacobian[vertex_index * 3 + 2, alpha_index] - d_c2) < 0.0001)

            shape = self.forward_pass(alphas)
            r = v_targets - shape
            r_flatten = r.flatten()
            # r_norm = np.linalg.norm(r_flatten)
            # print(r_norm)

            # <<methods for non-linear least squares problems>>, page 23
            A = np.matmul(jacobian.transpose(), jacobian)
            # print(A)
            b = -np.matmul(jacobian.transpose(), r_flatten)
            # print(b)
            gh1 = scipy.linalg.solve(A, b)
            # print(gh1)
            alphas += gh1
            if np.linalg.norm(gh1) < 0.00001:
                break

            # matrix inverse is slow
            # gh3 = np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ r_flatten
            # # print(gh3)
            # alphas -= gh3
            # if np.linalg.norm(gh3) < 0.00000001:
            #     break

        # print(alphas)
        # print('done')
        return alphas

    # solve_alphas_cwise_2: use w_sparse to encourage weights to stay sparse.
    def solve_alphas_cwise_2(self, in_alphas, v_targets, in_max_iter, w_fit, w_sparse):
        assert isinstance(w_fit, float)
        assert isinstance(w_sparse, float)
        alphas = np.copy(in_alphas)
        num_alphas = alphas.shape[0]
        # print(alphas)

        for i in range(in_max_iter):
            # jacobian1: fit part
            jacobian1 = self.calc_f_jacobian_cwise(alphas, v_targets) * w_fit

            # validation:
            # for vertex_index in range(self.get_num_vertices()):
            #     v_target = v_targets[vertex_index]
            #     for alpha_index in range(self.get_num_delta_shapes()):
            #         d_c0 = calc_partial_derv_numerical_cwise(self, alphas, alpha_index, vertex_index, v_target, 0)
            #         d_c1 = calc_partial_derv_numerical_cwise(self, alphas, alpha_index, vertex_index, v_target, 1)
            #         d_c2 = calc_partial_derv_numerical_cwise(self, alphas, alpha_index, vertex_index, v_target, 2)
            #         assert (abs(jacobian[vertex_index * 3 + 0, alpha_index] - d_c0) < 0.0001)
            #         assert (abs(jacobian[vertex_index * 3 + 1, alpha_index] - d_c1) < 0.0001)
            #         assert (abs(jacobian[vertex_index * 3 + 2, alpha_index] - d_c2) < 0.0001)

            # jacobian2: sparse part
            jacobian2 = np.identity(num_alphas) * w_sparse

            # full jacobian:
            jacobian = np.vstack((jacobian1, jacobian2))

            shape = self.forward_pass(alphas)
            r1 = (v_targets - shape) * w_fit
            r1_flatten = r1.flatten()
            # r_norm = np.linalg.norm(r_flatten)
            # print(r_norm)
            r2 = alphas * w_sparse
            r_flatten = np.concatenate((r1_flatten, r2), axis=0)

            # <<methods for non-linear least squares problems>>, page 23
            A = np.matmul(jacobian.transpose(), jacobian)
            # print(A)
            b = -np.matmul(jacobian.transpose(), r_flatten)
            # print(b)
            gh1 = scipy.linalg.solve(A, b)
            # print(gh1)
            alphas += gh1
            if np.linalg.norm(gh1) < 0.00001:
                break

            # matrix inverse is slow
            # gh3 = np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ r_flatten
            # # print(gh3)
            # alphas -= gh3
            # if np.linalg.norm(gh3) < 0.00000001:
            #     break

        # print(alphas)
        # print('done')
        return alphas


# component_wise version:
def calc_partial_derv_numerical_cwise(in_model, in_alphas, alpha_index, vertex_index, v_target, comp_index):
    eps = 0.0001
    perb = np.zeros(in_alphas.shape)
    perb[alpha_index] = eps
    f1 = v_target - in_model.forward_pass(in_alphas - perb)[vertex_index]
    f2 = v_target - in_model.forward_pass(in_alphas + perb)[vertex_index]
    derv = (f2[comp_index] - f1[comp_index]) / (2 * eps)
    # y1 = np.dot(f1[comp_index], f1[comp_index])
    # y2 = np.dot(f2[comp_index], f2[comp_index])
    # derv = (y2 - y1) / (2 * eps)
    return derv
