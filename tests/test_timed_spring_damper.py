import matplotlib.pyplot as plt
from simple_spring_damper_exact import *


def test_main():
    dt = 1.0 / 30
    x0 = 0.0
    x_goal = 1.0
    iter_count = 100

    halflife = 1
    xs1 = generate_timed_curve(x0, x_goal, 1.0, halflife, dt, iter_count)
    xs2 = generate_timed_curve(x0, x_goal, 2.0, halflife, dt, iter_count)
    xs3 = generate_timed_curve(x0, x_goal, 4.0, halflife, dt, iter_count)

    ts = [0.0] * (iter_count + 1)
    for i in range(len(ts)):
        ts[i] = dt * i

    # for i in range(count):
    #     plt.plot(ts, xss[i])

    plt.plot(ts, xs1, 'r')
    plt.plot(ts, xs2, 'b')
    plt.plot(ts, xs3, 'g')
    plt.show()


if __name__ == '__main__':
    test_main()



