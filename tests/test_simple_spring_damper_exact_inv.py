import matplotlib.pyplot as plt
from simple_spring_damper_exact_inv import *


def test_main():
    new_x, new_v = simple_spring_damper_exact_inv(0.0, 0.0, 1.0, 2.0, 1.0/30)

    dt = 1.0/30
    x_goal = 1.0
    x0 = 0.0

    iter_count = 99
    xs1 = generate_curve_inv(x0, x_goal, 8.0, dt, iter_count)
    xs2 = generate_curve_inv(x0, x_goal, 4.0, dt, iter_count)
    xs3 = generate_curve_inv(x0, x_goal, 2.0, dt, iter_count)
    xs4 = generate_curve_inv(x0, x_goal, 1.0, dt, iter_count)
    xs5 = generate_curve_inv(x0, x_goal, 0.5, dt, iter_count)

    ts = [0.0] * (iter_count + 1)
    for i in range(len(ts)):
        ts[i] = dt * i

    # plotting:
    assert len(ts) == len(xs1)
    plt.plot(ts, xs1, 'r')
    plt.plot(ts, xs2, 'c')
    plt.plot(ts, xs3, 'y')
    plt.plot(ts, xs4)
    plt.plot(ts, xs5)
    plt.show()


if __name__ == '__main__':
    test_main()