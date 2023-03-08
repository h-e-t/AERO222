from math import exp, tan, cos 

def f(x): 
    return tan(x) + exp(x) - x ** 2

def fp(x):\
    return 1 / cos(x)**2 + exp(x) - 2 * x

# 1A
def newton1A(f, fp):
    resT = 10 ** -8
    x0 = 0
    xold = 0
    res = f(x0)

    count = 0
    while abs(res) > resT:
        xnew = xold - f(xold)/fp(xold)
        res = f(xnew)
        print(xnew)
        xold = xnew
        count += 1

newton1A(f, fp)