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


def timed_spring_damper_exact_inv(x, v, xi, x_goal, t_goal, halflife_inv, dt, apprehension=2.0):
    min_time = max(t_goal, dt)
    v_goal = (x_goal - xi) / min_time
    t_goal_future = dt + apprehension * (1.0 / halflife_inv)
    if t_goal_future < t_goal:
        x_goal_future = xi + v_goal * t_goal_future
    else:
        x_goal_future = x_goal
    new_x, new_v = simple_spring_damper_exact_inv(x, v, x_goal_future, halflife_inv, dt)
    new_xi = xi + v_goal * dt
    return new_x, new_v, new_xi


def generate_curve_inv(x0, x_goal, halflife_inv, dt, iter_count):
    xs = [x0]
    x = x0
    v = 0.0
    for i in range(iter_count):
        x, v = simple_spring_damper_exact_inv(x, v, x_goal, halflife_inv, dt)
        xs.append(x)
    return xs


def generate_timed_curve_inv(x0, x_goal, goal_time, halflife_inv, dt, iter_count):
    xs = [x0]
    x = x0
    v = 0.0
    xi = x
    t_goal = goal_time
    for i in range(iter_count):
        x, v, xi = timed_spring_damper_exact_inv(x, v, xi, x_goal, t_goal, halflife_inv, dt)
        xs.append(x)
        t_goal -= dt
    return xs
