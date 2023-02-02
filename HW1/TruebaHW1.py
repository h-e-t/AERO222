from math import factorial
from numpy import sin, cos 

def f(x):
    return 3 * x**2 * sin(x) - x * cos(x) + 4  

def fp(x):
    return 3 * x**2 * cos(x) + 7 * x * sin(x) - cos(x)

def regulaFalsi():
    a = -2
    b = 2

    eps = 1E-8
    limit = 1
    counter = 0 

    while abs(limit) > eps:
        counter += 1
        x_next = b - (f(b) * (b - a))/(f(b) - f(a))
        limit = f(x_next)/fp(x_next)
        print(x_next)

        if f(x_next) * f(a) > 0:
        # if x_next * a > 0:
            a = x_next
        else:
            b = x_next

        

    print(f(b)/fp(b))

# regulaFalsi()


def roundOffA():
    x = 1.32
    
    machP = 3.15 * x**3 - 2.11 * x**2 - 4.01 * x + 10.33
    print(f'The value of the function wiht machine precision is {machP}')
    print()

    threeP = round(3.15 * x**3, 3) - round(2.11 * x**2, 3) - round(4.01 * x, 3) + 10.33
    print(f'The value with rounding at each arithmetic operation is {threeP}')
    print()

    aError = abs(machP - threeP)
    rError = abs(machP - threeP)/machP
    print(f'The absolute error is {aError} and relative error is {rError}')
    
def roundOffB():
    #nesting takes fewer operations 
    x = 1.32

    machP = ((3.15 * x - 2.11) * x - 4.01) * x + 10.33

    threeP = round((round((round(3.15 * x, 3) - 2.11) * x, 3) - 4.01) * x, 3) + 10.33

    aError = abs(machP - threeP)
    rError = abs(machP - threeP)/machP

    print(f'The value of nesting with machine precision is {machP}')
    print()
    print(f'The value of nesting with rounding at each arithmetic operation is {threeP}')
    print()
    print(f'The absolute error is {aError} and relative error is {rError}')
    
def sum3A():
    k = 0
    sum = 0  
    while k <= 10:
        sum += (-5)**k / factorial(k)
        k += 1

    print(sum)

def sum3B(): 
    k = 0
    sum = 0 
    while k <= 10: 
        sum += 5**k / factorial(k)
        k += 1 
    num = 1 / sum

    print(num)


sum3A()
sum3B()