

def fast_negexp(x):
    return 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
