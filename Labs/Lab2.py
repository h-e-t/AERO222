import numpy as np
import matplotlib.pyplot as plt
# Fixed point method 
def g1(x): 
    return np.sqrt(2*x + 3)

def g2(x):
    return 3 / (x-2)




x = np.linspace(-10, 10, 500)

plt.plot(x[x > -3/2], g1(x), 'b-', label = 'g1')
plt.plot(x, g2(x), 'r-', label = 'g2')
# plt.plot(x, g1(x), 'g-', label = 'g3')

plt.plot(x, x, 'k--', label = 'g(x) = x')

plt.ylim([-6, 6])

plt.legend()
plt.show()



# def fixedPointIteration(func, dFunc, )