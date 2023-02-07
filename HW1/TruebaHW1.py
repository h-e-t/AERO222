from math import cos, sin, factorial
import matplotlib.pyplot as plt 


def f (x):
    return (3 * x**2 * sin(x) - x * cos(x) + 4)

def fp(x): 
    return (3 * x**2 * cos(x) + 7 * x * sin(x) - cos(x))

def bisection():
    #start 1
    a = -2
    #end 1
    b = 2 
    
    # counter 
    n = 0 
    error = 1

    # threshold 
    eps = 1 / 10**8

    # f(x_n)
    values = []

    
    iterations = []

    while error > eps : 
        # error = abs(b-a)
        c = a + (b - a) / 2

        error = abs(f(c)/fp(c))        

        Fval = f(c)

        if(f(a)<f(b)):
            if Fval < 0: 
                a = c 
            else:
                b = c
        else: 
            if Fval > 0: 
                a = c 
            else:
                b = c            
        n += 1
        
        values.append(f(c))
        iterations.append(n)
        
        # print(f'a = {round(a, 3)} and b = {round(b,3)}')
        # print("Error is {:.3e}".format(error))
        # print()


        # # print(error > eps)


    print(f'Bisection found root at {round(a, 9)} in {n} iterations with {error} error')
    # print(residuals)
    # print(numbers)

    # plotting(iterations, values, 'Bisection')

    return iterations, values, 'Bisection'

def secant(): 
    #start 
    a = -2
    #end 
    b = 2 
    
    # counter 
    n = 0 
    error = abs(b-a) 
    eps = 1 / 10**8
    values = []
    iterations = []

    x_1 = a
    x_2 = b

    while error > eps and n < 20:

        x_next = x_2 - (f(x_2) * (x_2-x_1)) / (f(x_2) - f(x_1)) 

        x_1 = x_2
        x_2 = x_next

        n += 1

        error = abs(f(x_next)/fp(x_next)) 

        values.append(f(x_next))
        iterations.append(n)

        # print(x_next)
    print(f'Secant found root at {round(x_next, 9)} in {n} iterations with {error} error')
    # plotting(iterations, values, 'Secant')
    return iterations, values, 'Secant'

def regulaFalsi():
    a = -2
    b = 2

    eps = 1E-8
    error = 1
    n = 0 
    
    values = []
    iteration = []
   
    while abs(error) > eps:
        n += 1

        x_next = b - (f(b) * (b - a))/(f(b) - f(a))
        error = f(x_next)/fp(x_next)
        # print(x_next)

        if f(x_next) * f(a) > 0:
        # if x_next * a > 0:
            a = x_next
        else:
            b = x_next

        values.append(f(x_next))
        iteration.append(n)
    
    print(f'Regula Falsi found root at {round(x_next, 9)} in {n} iterations with {error} error')

    # plotting(iteration, values, 'Regula Falsi')
    # print(values)
    # print(iteration)
    # print(counter)
    return iteration, values, 'Regula Falsi'  

def roundOffA():
    x = 1.32
    
    machP = 3.15 * x**3 - 2.11 * x**2 - 4.01 * x + 10.33
    print(f'The value of the function wiht machine precision is {machP}')
    print()

    threeP = round(3.15 * x**3, 3) - round(2.11 * x**2, 3) - round(4.01 * x, 3) + 10.33
    print(f'The value with rounding at each arithmetic operation is {threeP}')

    aError = abs(machP - threeP)
    rError = abs(machP - threeP)/machP
    print(f'The absolute error is {aError} and relative error is {rError}')
    print()
    
def roundOffB():
    #nesting takes fewer operations 
    x = 1.32

    machP = ((3.15 * x - 2.11) * x - 4.01) * x + 10.33

    threeP = round((round((round(3.15 * x, 3) - 2.11) * x, 3) - 4.01) * x, 3) + 10.33

    aError = abs(machP - threeP)
    rError = abs(machP - threeP)/machP

    print(f'The value of nesting with machine precision is {machP}')
    print(f'The value of nesting with rounding at each arithmetic operation is {threeP}')
    print(f'The absolute error is {aError} and relative error is {rError}')
    print()

realE = 6.7379 * 10**-3

def sum3A():
    k = 0
    sum = 0  

    iterations = []
    error = []

    while k < 10:
        sum += (-5)**k / factorial(k)
        k += 1

        iterations.append(k)
        error.append(abs(sum-realE))

        # print(sum)
    
    

    # plt.plot(iterations, summing)
    # plt.title('Sum 3A')
    # plt.show()

    print(f'The value of sum 3A after 10 iterations is {round(sum, 5)}')
    return iterations, error, 'Sum 3A'

def sum3B(): 
    k = 0
    sum = 0 

    num = 0
    iterations = []
    error = []
    while k < 10: 
        sum += 5**k / factorial(k)
        k += 1 
        num = 1 / sum
        iterations.append(k)
        error.append(abs(num - realE))

    print(f'The value of sum 3A after 10 iterations is {round(num, 5)}')
    # plt.semilogy(iterations, summing)
    # plt.title('Sum 3B')
    # plt.show()

    # print(f'')

    return iterations, error, 'Sum 3B'

print()
biSect = bisection()
sec = secant()
rF = regulaFalsi()

plt.plot(biSect[0], biSect[1], label = biSect[2])
plt.plot(sec[0], sec[1], label = sec[2])
plt.plot(rF[0], rF[1], label = rF[2])

plt.xlabel('Iteration')
plt.ylabel('F(x_n)')

plt.legend(loc = 'upper right')
plt.show()  
        
roundOffA()
roundOffB()

sumA = sum3A()
sumB = sum3B()

print('A reason for the second sum converging to the true value so much quicker is that it')
print('always has the same sign, meaning that every additional term gets it closer to the real value.')
print("The other sum flips across the x axis, meaning it doesn't always get closer and takes longer to converge.")

plt.semilogy(sumA[0], sumA[1],label = sumA[2])
plt.semilogy(sumB[0], sumB[1],label = sumB[2])

plt.xlabel('Iteration')
plt.ylabel('Error')

plt.legend(loc = 'center right')

plt.show()


