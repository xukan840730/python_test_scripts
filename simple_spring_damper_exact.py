from common import *


def halflife_to_damping(halflife, eps=1e-5):
    return (4.0 * 0.69314718056) / (halflife + eps)


def simple_spring_damper_exact(x, v, x_goal, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0
    j0 = x - x_goal
    j1 = v + j0 * y
    eydt = fast_negexp(y * dt)

    x = eydt * (j0 + j1 * dt) + x_goal
    v = eydt * (v - j1 * y * dt)
    return x, v


def double_spring_damper_exact(x, v, xi, vi, x_goal, halflife, dt):
    t1 = 0.5
    t2 = 1.0 - t1
    new_xi, new_vi = simple_spring_damper_exact(xi, vi, x_goal, t1 * halflife, dt)
    new_x, new_v = simple_spring_damper_exact(x, v, new_xi, t2 * halflife, dt)
    return new_x, new_v, new_xi, new_vi


def timed_spring_damper_exact(x, v, xi, x_goal, t_goal, halflife, dt, apprehension=2.0):
    min_time = max(t_goal, dt)
    v_goal = (x_goal - xi) / min_time
    t_goal_future = dt + apprehension * halflife
    if t_goal_future < t_goal:
        x_goal_future = xi + v_goal * t_goal_future
    else:
        x_goal_future = x_goal
    new_x, new_v = simple_spring_damper_exact(x, v, x_goal_future, halflife, dt)
    new_xi = xi + v_goal * dt
    return new_x, new_v, new_xi


def generate_curve(x0, x_goal, halflife, dt, iter_count):
    xs = [x0]
    x = x0
    v = 0.0
    for i in range(iter_count):
        x, v = simple_spring_damper_exact(x, v, x_goal, halflife, dt)
        xs.append(x)
    return xs


def generate_curve_double(x0, x_goal, halflife, dt, iter_count):
    xs = [x0]
    x = x0
    v = 0.0
    xi = x
    vi = v
    for i in range(iter_count):
        x, v, xi, vi = double_spring_damper_exact(x, v, xi, vi, x_goal, halflife, dt)
        xs.append(x)
    return xs


def generate_timed_curve(x0, x_goal, goal_time, halflife, dt, iter_count):
    xs = [x0]
    x = x0
    v = 0.0
    xi = x
    t_goal = goal_time
    for i in range(iter_count):
        x, v, xi = timed_spring_damper_exact(x, v, xi, x_goal, t_goal, halflife, dt)
        xs.append(x)
        t_goal -= dt
    return xs