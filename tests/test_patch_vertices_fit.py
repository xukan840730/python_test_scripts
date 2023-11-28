import numpy as np
import matplotlib.pyplot as plt


def np_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# plane3d: ax+by+cz+w=0
class Plane3D:
    def __init__(self):
        self.normal = None
        self.w = None

    @staticmethod
    def from_point_and_normal(point, normal):
        res = Plane3D()
        res.normal = np_normalize(normal)
        res.w = -1.0 * (point @ res.normal)  # dot_product
        return res

    # return origin projected onto plane
    def get_origin(self):
        return self.normal * -self.w

    # return point distance to plane
    def plane_dot(self, point):
        return self.normal[0] * point[0] + self.normal[1] * point[1] + self.normal[2] * point[2] + self.w

    def project_point(self, point):
        return point - self.plane_dot(point) * self.normal


def points_fit_plane(points):
    centroid = np.mean(points, axis=0)
    u, s, v = np.linalg.svd(points - centroid)
    normal = v[-1]
    return Plane3D.from_point_and_normal(centroid, normal)


def test_main():
    with open('test_patch_892.npy', 'rb') as f:
        points = np.load(f)
        print(points)

        fit_plane = points_fit_plane(points)
        print(fit_plane)

        points_proj = np.zeros(points.shape)
        for i_vertex in range(points.shape[0]):
            points_proj[i_vertex] = fit_plane.project_point(points[i_vertex])

        centroid_proj = np.mean(points_proj, axis=0)
        dists2 = np.zeros(points.shape[0])
        # distance to centroid_proj
        for i_vertex in range(dists2.shape[0]):
            dists2[i_vertex] = np.linalg.norm(points_proj[i_vertex] - centroid_proj)

        best_index = np.argmin(dists2)
        best_center_pos = points_proj[best_index]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        points_x = points[:, 0]
        points_y = points[:, 1]
        points_z = points[:, 2]
        ax.scatter(points_x, points_y, points_z, color='b')

        points_proj_x = points_proj[:, 0]
        points_proj_y = points_proj[:, 1]
        points_proj_z = points_proj[:, 2]
        ax.scatter(points_proj_x, points_proj_y, points_proj_z, color='c')

        best_point_x = [best_center_pos[0]]
        best_point_y = [best_center_pos[1]]
        best_point_z = [best_center_pos[2]]
        ax.scatter(best_point_x, best_point_y, best_point_z, color='r')

        # X, Y = np.meshgrid(np.arange(-3.0, -1.0, 0.4),
        #                    np.arange(-11.0, -8.0, 0.4))
        # Z = np.zeros(X.shape)
        # for r in range(X.shape[0]):
        #     for c in range(X.shape[1]):
        #         Z[r, c] = -(fit_plane.normal[0] * X[r, c] + fit_plane.normal[1] * Y[r, c] + fit_plane.w) / fit_plane.normal[2]
        #
        # ax.plot_wireframe(X, Y, Z, color='k')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()


if __name__ == '__main__':
    test_main()
