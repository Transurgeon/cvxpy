import time

import numpy as np

import cvxpy as cp

timings = {}

m = 5
for n in np.logspace(6, 7, 1).astype(int):
    timings[n] = {}
    print(n)
    np.random.seed(1)
    A = np.random.randn(m, n)
    c = np.random.randn(1, n)

    # Construct the problem.
    x = cp.Variable((n, 1), nonneg=True)
    objective = cp.Minimize(c @ x)
    constraints = [A @ x <= 0, x <= 1]
    prob = cp.Problem(objective, constraints)

    start = time.time()
    prob.get_problem_data(canon_backend='RUST', solver=cp.SCS)
    timings[n]['RUST'] = time.time() - start

    start = time.time()
    prob.get_problem_data(canon_backend='CPP', solver=cp.SCS)
    timings[n]['CPP'] = time.time() - start

    start = time.time()
    prob.get_problem_data(canon_backend='SCIPY', solver=cp.SCS)
    timings[n]['SCIPY'] = time.time() - start
    print(timings)
