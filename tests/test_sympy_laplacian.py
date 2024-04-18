from sympy import symbols
from sympy.matrices import Matrix


def test_func():
    d0, d1, d2, d3 = symbols('d0, d1, d2, d3')

    w0, w1, w2, w3 = symbols('w0, w1, w2, w3')

    w01, w02, w03 = symbols('w01, w02, w03')
    w10, w12, w13 = symbols('w10, w12, w13')
    w20, w21, w23 = symbols('w20, w21, w23')
    w30, w31, w32 = symbols('w30, w31, w32')

    ks, kb = symbols('ks, kb')

    # do operations 1:
    lap_d0 = w0 * (w01 * (d1 - d0) + w02 * (d2 - d0) + w03 * (d3 - d0))
    lap_d1 = w1 * (w10 * (d0 - d1) + w12 * (d2 - d1))
    lap_d2 = w2 * (w20 * (d0 - d2) + w21 * (d1 - d2) + w23 * (d3 - d2))
    lap_d3 = w3 * (w30 * (d0 - d3) + w32 * (d2 - d3))

    lap_sqr_d0 = w0 * (w01 * (lap_d1 - lap_d0) + w02 * (lap_d2 - lap_d0) + w03 * (lap_d3 - lap_d0))
    lap_sqr_d1 = w1 * (w10 * (lap_d0 - lap_d1) + w12 * (lap_d2 - lap_d1))
    lap_sqr_d2 = w2 * (w20 * (lap_d0 - lap_d2) + w21 * (lap_d1 - lap_d2) + w23 * (lap_d3 - lap_d2))
    lap_sqr_d3 = w3 * (w30 * (lap_d0 - lap_d3) + w32 * (lap_d2 - lap_d3))

    res0 = -ks * lap_d0 + kb * lap_sqr_d0
    print(res0)
    res1 = -ks * lap_d1 + kb * lap_sqr_d1
    print(res1)
    res2 = -ks * lap_d2 + kb * lap_sqr_d2
    print(res2)
    res3 = -ks * lap_d3 + kb * lap_sqr_d3
    print(res3)
    res = [res0, res1, res2, res3]

    # do operations 2:
    D = Matrix([[w0, 0, 0, 0], [0, w1, 0, 0], [0, 0, w2, 0], [0, 0, 0, w3]])
    M = Matrix([[-w01-w02-w03, w01, w02, w03], [w10, -w10-w12, w12, 0], [w20, w21, -w20-w21-w23, w23], [w30, 0, w32, -w30-w32]])
    L = D * M
    L_sqr = L * L
    A = -ks * L + kb * L_sqr
    d_vec = Matrix([d0, d1, d2, d3])
    res_vec = A * d_vec
    print(res_vec[0])
    print(res_vec[1])
    print(res_vec[2])
    print(res_vec[3])

    # evaluation:
    subs = {
        d0: 0.11, d1: 0.12, d2: 0.13, d3: 0.14,
        ks: 1.0, kb: 1.0,
        w0: 1.1, w1: 0.9, w2: 0.8, w3: 1.2,
        w01: 0.78, w02: 0.79, w03: 0.80,
        w10: 0.81, w12: 0.82, w13: 0.83,
        w20: 0.84, w21: 0.84, w23: 0.86,
        w30: 0.87, w31: 0.88, w32: 0.89,
    }

    for i in range(4):
        valI = res[i].evalf(20, subs=subs)
        print(valI)
        valJ = res_vec[i].evalf(20, subs=subs)
        print(valJ)


if __name__ == '__main__':
    test_func()
