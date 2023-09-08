from common import *


def halflife_inv_to_damping(halflife_inv):
    return (4.0 * 0.69314718056) * halflife_inv


def simple_spring_damper_exact_inv(x, v, x_goal, halflife_inv, dt):
    y = halflife_inv_to_damping(halflife_inv) / 2.0
    j0 = x - x_goal
    j1 = v + j0 * y
    eydt = fast_negexp(y * dt)

    x = eydt * (j0 + j1 * dt) + x_goal
    v = eydt * (v - j1 * y * dt)
    return x, v


def generate_curve_inv(x0, x_goal, halflife_inv, dt, iter_count):
    xs = [x0]
    x = x0
    v = 0.0
    for i in range(iter_count):
        x, v = simple_spring_damper_exact_inv(x, v, x_goal, halflife_inv, dt)
        xs.append(x)
    return xs

