import matplotlib.pyplot as plt


def halflife_inv_to_damping(halflife_inv):
    return (4.0 * 0.69314718056) * halflife_inv


def fast_negexp(x):
    return 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)


def simple_spring_damper_exact_2(x, v, x_goal, halflife_inv, dt):
    y = halflife_inv_to_damping(halflife_inv) / 2.0
    j0 = x - x_goal
    j1 = v + j0 * y
    eydt = fast_negexp(y * dt)

    x = eydt * (j0 + j1 * dt) + x_goal
    v = eydt * (v - j1 * y * dt)
    return x, v


def generate_curve(x0, x_goal, halflife_inv, dt, iter_count):
    xs = [x0]
    x = x0
    v = 0.0
    for i in range(iter_count):
        x, v = simple_spring_damper_exact_2(x, v, x_goal, halflife_inv, dt)
        xs.append(x)
    return xs


dt = 1.0/30
x_goal = 1.0
x0 = 0.0

iter_count = 99
xs1 = generate_curve(x0, x_goal, 2.0, dt, iter_count)
xs2 = generate_curve(x0, x_goal, 1.0, dt, iter_count)
xs3 = generate_curve(x0, x_goal, 0.5, dt, iter_count)

ts = [0.0] * (iter_count + 1)
for i in range(len(ts)):
    ts[i] = dt * i

# plotting:
assert len(ts) == len(xs1)
plt.plot(ts, xs1, 'r')
plt.plot(ts, xs2, 'c')
plt.plot(ts, xs3, 'y')
plt.show()
