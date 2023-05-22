import matplotlib.pyplot as plt
from spring_tracker import *


def halflife_to_damping(halflife, eps=1e-5):
    return (4.0 * 0.69314718056) / (halflife + eps)


def fast_negexp(x):
    return 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)


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


def generate_curve(x0, x_goal, halflife, dt, iter_count):
    xs = [x0]
    x = x0
    v = 0.0
    for i in range(iter_count):
        x, v = simple_spring_damper_exact(x, v, x_goal, halflife, dt)
        xs.append(x)
    return xs


def generate_curve_old(x0, x_goal, spring_const, mass, dt, iter_count):
    xs = [x0]
    x = x0
    v = 0.0
    for i in range(iter_count):
        x, v = track_common(v, x, x_goal,spring_const, mass, dt)
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


dt = 1.0/30
x_goal = 1.0
x0 = 0.0

iter_count = 99
test_halflife_inv = 2.0
test_halflife = 1.0 / test_halflife_inv
# xs1 = generate_curve(x0, x_goal, 2.0, dt, iter_count)
# xs2 = generate_curve(x0, x_goal, 1.0, dt, iter_count)
xs3 = generate_curve(x0, x_goal, 0.125, dt, iter_count)
xs4 = generate_curve(x0, x_goal, test_halflife, dt, iter_count)

ys1 = generate_curve_old(x0, x_goal, 4.0, 0.5, dt, iter_count)
# ys2 = generate_curve_old(x0, x_goal, 200.0, 0.5, dt, iter_count)

zs1 = generate_curve_double(x0, x_goal, test_halflife, dt, iter_count)

ts = [0.0] * (iter_count + 1)
for i in range(len(ts)):
    ts[i] = dt * i

# plotting:
# assert len(ts) == len(xs1)
# plt.plot(ts, xs1, 'r')
# plt.plot(ts, xs2, 'c')
# plt.plot(ts, xs3, 'y')
plt.plot(ts, xs4, 'g')

# plt.plot(ts, ys1, 'r')
# plt.plot(ts, ys2)
plt.plot(ts, zs1, 'r')
plt.show()
