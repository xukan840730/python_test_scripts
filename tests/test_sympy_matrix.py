from sympy.matrices import Matrix
from sympy import symbols, diff


def test_func():
    w1 = symbols('w1')
    w2 = symbols('w2')
    w3 = symbols('w3')

    w_hat = Matrix([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
    m = Matrix.eye(3)
    for i in range(1, 10):
        m *= w_hat
        print(f'{i}: {m}')
        w1_derv = diff(m, w1)
        print(f'w1_derv: {w1_derv}')


if __name__ == "__main__":
    test_func()
