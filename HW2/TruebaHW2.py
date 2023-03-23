from math import exp, cos 
import numpy as np 
import matplotlib.pyplot as plt 

def f(x): 
    return np.arctan(x) + exp(x) - x ** 2

def fp(x):
    return 1 / (1+x**2) + np.exp(x) - 2 * x

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
        # print(xnew)
        xold = xnew
        count += 1

    print(f'(1A) The final residual is {f(xnew)} after {count} iterations')

def newton1B(f, fp):
    xold = 0
    x = xold - f(xold)/fp(xold)
    xnew = x - f(x)/fp(x)
    count = 0 
    error = 0

    while abs(xnew - x) < abs(x-xold):
        xold = x 
        x = xnew
        xnew = xnew - f(xnew)/fp(xnew)

        count += 1

        error = abs(f(xnew)/fp(xnew))
        # print(abs(xnew - x) < abs(x-xold))

    # print(count)
    # print(error)
    print(f'(1B) The final recorded error was {error} after {count} iterations.')


def ROC(fp):
    x = np.linspace(-10,10,500)

    plt.plot(x,fp(x),'b-',label=r'$g^\prime(x)$')
    plt.plot(x,-1*np.ones_like(x),'k--',label=r'$g(x) = \pm 1$')
    plt.plot(x,np.ones_like(x),'k--')
    plt.xlim([-10,10])
    plt.ylim([-3,3])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$g^\prime(x)$')
    plt.title('Range of convergence for the Newton method')
    plt.legend()
    plt.show()

print()
newton1A(f, fp)
print()
newton1B(f, fp)
ROC(fp)