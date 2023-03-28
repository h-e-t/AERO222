import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

# #CODING PROBLEM 1
# print("Problem 1")
# x = np.linspace(-0.5,1.5,100)
# yTrue = np.cos(3*x)
# yMeasured = yTrue + np.random.normal(0, 0.12, 100)

# # linspace of 100 data points for xk, yk from -.5 to 1.5
# # write for function cos(3x)

# # A matrix
# A_1 = np.vstack([np.ones(len(x)),x,2*x**2 - 1,4*x**3 - 3*x]).T
# A_2 = np.vstack([np.cos(x)**2, 1 - 2 * np.sin(x), np.cos(3*x) * np.sin(x), (3 - x) / (3 + x)]).T

# # reshape y
# yMeasured_1 = yMeasured.reshape(A_1.shape[0],1)
# yMeasured_2 = yMeasured.reshape(A_2.shape[0],1)

# # solve least-squares
# #c = np.linalg.inv(A.T@A) @ A.T @ yMeasured
# c_1 = np.linalg.pinv(A_1) @ yMeasured_1
# c_2 = np.linalg.pinv(A_2) @ yMeasured_2

# #linear algebra psuedo-inverse is already built in using the line above
# yHat_1 = c_1[0] + c_1[1]*x + c_1[2]*(2*x**2 - 1) + c_1[3]*(4*x**3 - 3*x)
# yHat_2 = c_2[0] * np.cos(x)**2 + c_2[1]*(1 - 2 * np.sin(x)) + c_2[2] * np.cos(3*x) * np.sin(x) + c_2[3]*(3 - x) / (3 + x)

# L1_y1 = LA.norm(yHat_1 , ord = 1)
# L2_y1 = LA.norm(yHat_1 , ord = 2)
# L_inf_y1 = LA.norm(yHat_1 , ord = np.inf)
# print("L1 norm for y1: ", L1_y1)
# print("L2 norm for y1: ", L2_y1)
# print("L∞ norm for y1: ", L_inf_y1)

# print()

# L1_y2 = LA.norm(yHat_2 , ord = 1)
# L2_y2 = LA.norm(yHat_2 , ord = 2)
# L_inf_y2 = LA.norm(yHat_2 , ord = np.inf)
# print("L1 norm for y2: ", L1_y2)
# print("L2 norm for y2: ", L2_y2)
# print("L∞ norm for y2: ", L_inf_y2)


# # plot
# plt.plot(x,yMeasured,'r.',label='data points')
# plt.plot(x,yHat_1,'b-',label='linear fit 1')
# plt.plot(x,yHat_2,'g-',label='linear fit 2')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Linear Regression')
# plt.legend()
# plt.savefig('Problem 1 - Linear Regression.jpg', dpi = 1200)
# plt.show()

# print()

# #CODING PROBLEM 3
# print('Problem 3')
# t = np.linspace(0,5,100)
# yTrue = np.exp(-t) * np.cos(t)
# yMeasured = yTrue + t**1/2 * np.random.normal(0, 0.05, 100)

# w_t = 1/(1 + 0.1**(2.5 - t))
# diag_w = np.diag(w_t)

# #supplemental notes
# plt.plot(t,w_t,'y.')
# plt.title('Weight Function')
# plt.xlabel('x')
# plt.ylabel('w(t)')
# plt.savefig('Problem 3 - Weight Function.jpg', dpi = 1200)
# plt.show()

# print('There is more noise as the data set increases on the x-axis, so it makes sense to assign more weight to the data long those early intervals.')


# A = np.vstack([np.ones(len(x)),x,2*x**2 - 1,4*x**3 - 3*x]).T
# yMeasured = yMeasured.reshape(A.shape[0],1)
# c = np.linalg.inv(A.T @ diag_w @ A) @ A.T @ diag_w @ yMeasured

# yHat = c[0] + c[1]*x + c[2]*(2*x**2 - 1) + c[3]*(4*x**3 - 3*x)
# plt.plot(t,yTrue,'g-',label = 'x(t) Function')
# plt.plot(t,yMeasured,'r.',label = 'Data Points with Noise')
# plt.plot(t,yHat,'b--', label = '$\hat{x}$(t) Function')
# plt.xlabel('x')
# plt.ylabel('x(t)')
# plt.title('$\hat{x}$(t) and x(t) with Corrupted Data')
# plt.legend()
# plt.savefig('Problem 3 - xhat(t) and x(t) with Corrupted Data.jpg', dpi = 1200)
# plt.show()

# print()

#CODING PROBLEM 4
print('Problem 4')
# Function
def f(x,B):
    return 12 - B[1] * x ** 2 - np.exp(-(B[3] * x ** 2)) * np.sin(B[2] * x) + B[0]*(B[1] * B[3] - B[2] * x)

# True parameters
B_true = np.array([-2,1,-1,5])

# x-values
x = np.linspace(-1,1,100)
# Noise
noise = np.random.normal(0,0.1,100)
f_true =  f(x,B_true)
f_meas =  f_true + noise

# plt.plot(x,f_true,'b-',label='truth')
# plt.plot(x,f_meas,'r.',label='data')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('Nonlinear Function and Simulated Data')
# plt.legend()
# plt.savefig('Problem 4 - Nonlinear Function and Simulated Data', dpi = 1200)
# plt.show()



# Jacobian
def J(x,B):
    a, b, c, d = B
    return np.array([b * d - c * x, -(x**2 - a * d), -np.cos(c * x) * x * np.exp(-d * x**2), np.exp(-d*x**2) * x**2 * np.sin(c * x) + a * b]).T

B_0 = np.array([-2.2,0.3,-1,5.2],dtype=float) # Initial guess
maxIter = 1000 # Max iteration
tol = 1e-6 # Tolerance
res = f(x,B_0) - f(x,B_true) # Initial residual
norm = LA.norm(res,2)
norm_list = []
iterations = []

B_hist = np.array([B_0.copy()]) # Parameter history
for i in range(1,maxIter+1):
    B = B_0 - np.linalg.pinv(J(x,B_0)) @ res
    B_0 = B
    res = f(x,B_0) - f(x,B_true) # Residual
    resNorm = np.linalg.norm(res,2) # Normed residual
    norm_list.append(resNorm)
    iterations.append(i)

    B_hist = np.concatenate([B_hist,[B]],0) # Append current parameters

    # Convergence criteria
    if resNorm < tol:
        print(f'Converged after {i} iterations')
        print(f'Normed residual = {resNorm}')
        print(f'Estimated A = {B[0]}')
        print(f'Estimated B = {B[1]}')
        print(f'Estimated C = {B[2]}')
        print(f'Estimated D = {B[3]}')
        break

    # Failure to converge
    if i == maxIter:
        print(f'Could not converge within {maxIter} Iterations')

plt.plot(x,f_meas,'r.',label='data')
for i in range(1,len(B_hist)):
    plt.plot(x,f(x,B_hist[i]),label=f'i = {i}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Nonlinear Least-Squares Fit')
plt.legend()
plt.savefig('Problem 4 - Nonlinear Least-Squares Fit.jpg', dpi = 1200)
plt.show()

plt.plot(iterations,norm_list)
plt.title('L2 Norm of the Residual Vector')
plt.xlabel('Iterations')
plt.ylabel('L2 Norm')
plt.savefig('Problem 4 - L2 Norm of the Residual Vector.jpg', dpi = 1200)
plt.show()