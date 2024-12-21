"""
======================
Portfolio Optimization
======================

In this example we show how to do portfolio optimization using CVXPY. 
We begin with the basic definitions. In portfolio optimization we have 
some amount of money to invest in any of n different assets
"""

import numpy as np

import cvxpy as cp

# Generate data.
m = 20
n = 15
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
cost = cp.sum_squares(A @ x - b)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)
print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)
