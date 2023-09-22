import matplotlib.pyplot as plt
from spring_tracker.simple_spring_damper_exact import *


def test_main():
    new_x, new_v = simple_spring_damper_exact(0.0, 0.0, 1.0, 1 / 2.0, 1.0 / 30)
    new_x, new_v, new_xi, new_vi = double_spring_damper_exact(0, 0, 0, 0, 1, 1.0 / 2.0, 1.0 / 30)

    dt = 1.0 / 30
    x_goal = 1.0
    x0 = 0.0
    # x_goal = -500
    # x0 = 500
    x0 = 0.0
    x_goal = 180.0

    iter_count = 40
    test_halflife_inv = 3.0
    test_halflife = 1.0 / test_halflife_inv
    # xs1 = generate_curve(x0, x_goal, 2.0, dt, iter_count)
    # xs2 = generate_curve(x0, x_goal, 1.0, dt, iter_count)
    xs3 = generate_curve(x0, x_goal, 0.125, dt, iter_count)
    xs4 = generate_curve(x0, x_goal, test_halflife, dt, iter_count)

    # ys1 = generate_curve_old(x0, x_goal, 4.0, 0.5, dt, iter_count)
    # ys2 = generate_curve_old(x0, x_goal, 200.0, 0.5, dt, iter_count)

    zs1 = generate_curve_double(x0, x_goal, 1.0 / 3.5, dt, iter_count)

    ts = [0.0] * (iter_count + 1)
    for i in range(len(ts)):
        ts[i] = dt * i

    # plotting:
    # assert len(ts) == len(xs1)
    # plt.plot(ts, xs1, 'r')
    # plt.plot(ts, xs2, 'c')
    # plt.plot(ts, xs3, 'y')
    # plt.plot(ts, xs4, 'g')

    # plt.plot(ts, ys1, 'r')
    # plt.plot(ts, ys2)
    plt.plot(ts, zs1, 'r')

    # zs2 = [
    # 500.00000000,
    # 498.99078428,
    # 494.22978905,
    # 482.55646368,
    # 461.81722354,
    # 431.21699594,
    # 391.16878027,
    # 342.95734717,
    # 288.38020542,
    # 229.44023926,
    # 168.11530950,
    # 106.20483805,
    # 45.24200044,
    # -13.54353008,
    # -69.22703800,
    # -121.16706517,
    # -168.97051285,
    # -212.45187764,
    # -251.59167717,
    # -286.49729513,
    # -317.36809259,
    # -344.46563934,
    # -368.08924357,
    # -388.55652906,
    # -406.18856238,
    # -421.29891402,
    # -434.18600694,
    # -445.12812949,
    # -454.38054605,
    # -462.17421057,
    # -468.71566452,
    # -474.18777567,
    # -478.75104228,
    # -482.54524765,
    # -485.69130135,
    # -488.29314585,
    # -490.43964233,
    # -492.20637621,
    # -493.65734481,
    # ]
    #
    # plt.plot(ts[0:39], zs2, 'c')

    plt.show()


if __name__ == '__main__':
    test_main()